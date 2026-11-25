"""Build expanded 8-city × 12-template (96-task) real OpenStreetMap global GIS benchmark (`data/osm_global/`).

Directly addresses external-validity (M5) and non-invariant task boundary (M1) critiques:
  - 8 real cities spanning Equatorial (Singapore, Quito), Southern Hemisphere (Sydney),
    Near-Antimeridian Pacific (Suva), High-Latitude (Helsinki, Reykjavik), Arctic Circle (Tromso),
    and High Arctic (Longyearbyen, 78.22N — Web Mercator area distortion > 24x!).
  - 12 distinct executable GIS task templates:
      * 9 metric templates (including 3 brand-new geometric/statistical tasks never in GSB v2:
        building perimeter km, 2nd-nearest POI median spacing, road convex-hull density)
      * 1 topological/spatial-join control (buildings intersecting road bounding boxes)
      * 2 EXPLICIT ABSOLUTE-POSITION NON-INVARIANT tasks (northernmost building centroid WGS84
        latitude; count of buildings strictly north of bbox midpoint latitude) — verifying that
        metamorphic training does NOT break tasks where spatial relocation is NOT answer-invariant!

Usage:
    nix-shell --run "python -m benchmarks.gsb_v2.build_osm_global --out data/osm_global"
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from .build_osm_slice import _aeqd, _bbox_for_city, fetch_osm_xml, parse_osm_xml

GLOBAL_CITIES = [
    # (slug, display_name, center_lat, center_lon, half_size_m)
    ("singapore", "Singapore (1.29N)", 1.2880, 103.8515, 320.0),
    ("quito", "Quito (-0.22S)", -0.2202, -78.5123, 320.0),
    ("sydney", "Sydney (-33.87S)", -33.8688, 151.2093, 300.0),
    ("wellington", "Wellington, NZ (-41.29S, 174.78E)", -41.2889, 174.7772, 340.0),
    ("helsinki", "Helsinki (60.17N)", 60.1695, 24.9410, 320.0),
    ("reykjavik", "Reykjavik (64.15N)", 64.1466, -21.9426, 360.0),
    ("tromso", "Tromso (69.65N)", 69.6492, 18.9553, 380.0),
    ("longyearbyen", "Longyearbyen (78.22N)", 78.2232, 15.6267, 550.0),
]


def _build_city_global_tasks(
    root: Path,
    slug: str,
    lat: float,
    lon: float,
    g_bldg: gpd.GeoDataFrame,
    g_road: gpd.GeoDataFrame,
    g_poi: gpd.GeoDataFrame,
) -> list[dict]:
    crs_m = _aeqd(lat, lon)
    b_m = g_bldg.to_crs(crs_m)
    r_m = g_road.to_crs(crs_m)
    p_m = g_poi.to_crs(crs_m)

    # 1-6: Core real-world metric templates
    tot_bldg_ha = float(b_m.geometry.area.sum()) / 1e4
    tot_road_km = float(r_m.geometry.length.sum()) / 1e3
    poi_buf_150 = unary_union(p_m.geometry.buffer(150.0))
    bldg_in_buf_ha = float(b_m.geometry.intersection(poi_buf_150).area.sum()) / 1e4
    large_bldg_ha = float(b_m.geometry.area[b_m.geometry.area > 300.0].sum()) / 1e4
    b_cent = b_m.geometry.centroid
    p_union = unary_union(p_m.geometry)
    median_dist_m = float(np.median([pt.distance(p_union) for pt in b_cent]))
    road_buf_30 = unary_union(r_m.geometry.buffer(30.0))
    poi_near_road_cnt = int(p_m.geometry.within(road_buf_30).sum())

    # 7-9: Brand-new unseen metric templates (never in GSB v2)
    tot_bldg_perim_km = float(b_m.geometry.length.sum()) / 1e3
    pts_xy = np.array([(pt.x, pt.y) for pt in p_m.geometry], dtype=np.float64)
    if len(pts_xy) >= 2:
        dmat = np.hypot(pts_xy[:, None, 0] - pts_xy[None, :, 0], pts_xy[:, None, 1] - pts_xy[None, :, 1])
        np.fill_diagonal(dmat, np.inf)
        nn_other_median_m = float(np.median(np.min(dmat, axis=1)))
    else:
        nn_other_median_m = 0.0
    road_hull_km2 = float(unary_union(r_m.geometry).convex_hull.area) / 1e6
    road_density_km_per_km2 = float(tot_road_km / max(road_hull_km2, 1e-9))

    # 10: Topological / non-projection control
    road_union_deg = unary_union(g_road.geometry)
    bldg_touch_or_cross_road = int(g_bldg.geometry.intersects(road_union_deg).sum())

    # 11-12: Explicit ABSOLUTE-POSITION (non-relocation-invariant) tasks
    # Compute centroid in local AEQD then transform back to EPSG:4326 for clean ellipsoidal centroid lat
    b_cent_4326 = gpd.GeoSeries(b_m.geometry.centroid, crs=crs_m).to_crs("EPSG:4326")
    # Note: for small 300-550m urban blocks, WGS84 centroid lat vs projected centroid lat agrees within <1e-5 deg
    lats = np.array([pt.y for pt in b_cent_4326], dtype=np.float64)
    northernmost_lat = float(np.max(lats))
    minx, miny, maxx, maxy = g_bldg.total_bounds
    mid_lat = 0.5 * (float(miny) + float(maxy))
    north_half_cnt = int(np.sum(lats > mid_lat))

    # Web Mercator distortion ratio for reporting
    b_wm = g_bldg.to_crs("EPSG:3857")
    wm_ratio = float(b_wm.geometry.area.sum() / max(b_m.geometry.area.sum(), 1e-9))

    task_specs = [
        (
            "osm_building_area_ha", "metric", "float", tot_bldg_ha, ["buildings.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`). "
            "Write a Python script that computes the total footprint area of all buildings in hectares (ha). Print only the final number.",
        ),
        (
            "osm_road_length_km", "metric", "float", tot_road_km, ["roads.gpkg"],
            "The current directory contains `roads.gpkg` (line road segments with columns `osm_id`, `highway_class`). "
            "Write a Python script that computes the total length of all road segments in kilometres (km). Print only the final number.",
        ),
        (
            "osm_poi_buffer_building_area", "metric", "float", bldg_in_buf_ha, ["buildings.gpkg", "pois.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`) and "
            "`pois.gpkg` (point features with columns `osm_id`, `category`). "
            "Write a Python script that creates a 150-metre buffer around every POI, takes the union of all buffers, "
            "intersects each building footprint with that union, and computes the total intersected building area in hectares (ha). "
            "Print only the final number.",
        ),
        (
            "osm_large_building_area_ha", "metric", "float", large_bldg_ha, ["buildings.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`). "
            "Write a Python script that selects all buildings whose footprint area is strictly greater than 300 square metres "
            "and computes their combined area in hectares (ha). Print only the final number.",
        ),
        (
            "osm_nearest_poi_median_dist", "metric", "float", median_dist_m, ["buildings.gpkg", "pois.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`) and "
            "`pois.gpkg` (point features with columns `osm_id`, `category`). "
            "Write a Python script that computes the Euclidean distance (in metres) from each building's centroid to its nearest POI, "
            "and prints the median distance across all buildings. Print only the final number.",
        ),
        (
            "osm_road_buffer_poi_count", "metric", "count", float(poi_near_road_cnt), ["roads.gpkg", "pois.gpkg"],
            "The current directory contains `roads.gpkg` (line road segments with columns `osm_id`, `highway_class`) and "
            "`pois.gpkg` (point features with columns `osm_id`, `category`). "
            "Write a Python script that counts how many POIs lie within 30 metres of at least one road segment. "
            "Print only the final integer.",
        ),
        (
            "osm_building_perimeter_km", "metric_new", "float", tot_bldg_perim_km, ["buildings.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`). "
            "Write a Python script that computes the total perimeter (boundary length) of all building polygons in kilometres (km). "
            "Print only the final number.",
        ),
        (
            "osm_poi_nn_spacing_m", "metric_new", "float", nn_other_median_m, ["pois.gpkg"],
            "The current directory contains `pois.gpkg` (point features with columns `osm_id`, `category`). "
            "Write a Python script that computes, for every POI, the distance in metres to its nearest OTHER POI "
            "(excluding itself), and prints the median of those nearest-neighbor distances in metres. Print only the final number.",
        ),
        (
            "osm_road_hull_density", "metric_new", "float", road_density_km_per_km2, ["roads.gpkg"],
            "The current directory contains `roads.gpkg` (line road segments with columns `osm_id`, `highway_class`). "
            "Write a Python script that computes the road network density as total road length (in km) divided by the area (in km^2) "
            "of the convex hull enclosing all road geometries. Print only the final number.",
        ),
        (
            "osm_bldg_road_intersects_cnt", "control", "count", float(bldg_touch_or_cross_road), ["buildings.gpkg", "roads.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints) and `roads.gpkg` (line road segments). "
            "Write a Python script that counts how many building polygons geometrically intersect at least one road segment. "
            "Print only the final integer.",
        ),
        (
            # "coordinate" = absolute tolerance in degrees (see benchmarks.gsb.harness.matches);
            # a 1% relative band would accept any latitude inside the city.
            "osm_northernmost_bldg_lat", "abs_pos", "coordinate", northernmost_lat, ["buildings.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints in WGS84 / EPSG:4326). "
            "Write a Python script that finds the maximum WGS84 latitude (in decimal degrees, positive north / negative south) "
            "among all building centroids. Print only the final number.",
        ),
        (
            "osm_north_half_bldg_count", "abs_pos", "count", float(north_half_cnt), ["buildings.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints in WGS84 / EPSG:4326). "
            "Let `mid_lat` be the midpoint between the minimum and maximum latitude of `buildings.gpkg`'s total bounding box in WGS84. "
            "Write a Python script that counts how many buildings have a centroid latitude strictly greater than `mid_lat`. "
            "Print only the final integer.",
        ),
    ]

    layers_map = {"buildings.gpkg": g_bldg, "roads.gpkg": g_road, "pois.gpkg": g_poi}
    tasks = []
    for fam, category, ans_type, truth, need_files, prompt in task_specs:
        tid = f"{fam}-{slug}-test-0000"
        vdir = root / tid / "low"
        vdir.mkdir(parents=True, exist_ok=True)
        files_dict = {}
        for fn in need_files:
            layers_map[fn].to_file(vdir / fn, driver="GPKG")
            files_dict[Path(fn).stem] = fn
        tasks.append({
            "task_id": tid,
            "family": f"{fam}__{slug}",
            "base_family": fam,
            "task_category": category,
            "city": slug,
            "group": category,
            "split": "test",
            "answer": ans_type,
            "files": files_dict,
            "prompt": prompt,
            "wm_area_ratio": round(wm_ratio, 4),
            "variants": {
                "low": {"dir": f"{tid}/low", "lat": round(lat, 5), "lon": round(lon, 5), "truth": truth}
            },
        })
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 8-city x 12-template global real OSM GIS benchmark")
    ap.add_argument("--out", default="data/osm_global")
    a = ap.parse_args()

    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    all_tasks: list[dict] = []
    for slug, disp, lat, lon, half_m in GLOBAL_CITIES:
        bbox = _bbox_for_city(lat, lon, half_m)
        print(f"Fetching real OSM extract for {disp} bbox={[round(x, 4) for x in bbox]}...", flush=True)
        xml_bytes = fetch_osm_xml(bbox, timeout=45.0)
        g_bldg, g_road, g_poi = parse_osm_xml(xml_bytes)
        print(f"  parsed buildings={len(g_bldg)}, roads={len(g_road)}, pois={len(g_poi)}", flush=True)
        if len(g_bldg) < 8 or len(g_road) < 8 or len(g_poi) < 5:
            raise RuntimeError(f"Insufficient OSM features in {slug}: bldg={len(g_bldg)}, road={len(g_road)}, poi={len(g_poi)}")
        city_tasks = _build_city_global_tasks(root, slug, lat, lon, g_bldg, g_road, g_poi)
        print(f"  built {len(city_tasks)} tasks | Web Mercator area distortion = {city_tasks[0]['wm_area_ratio']:.3f}x", flush=True)
        all_tasks.extend(city_tasks)
        time.sleep(1.5)

    with open(root / "tasks.jsonl", "w") as fh:
        for t in all_tasks:
            fh.write(json.dumps(t) + "\n")
    print(f"Wrote {len(all_tasks)} real global OSM tasks to {root}")


if __name__ == "__main__":
    main()

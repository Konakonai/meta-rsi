"""Build real OpenStreetMap 4-latitude urban GIS transfer slice (`data/osm_eval/`).

Fetches compact real-world urban extracts from the official OpenStreetMap v0.6 map API across
four latitudes — Singapore (1.29N), Xuzhou (34.26N), Helsinki (60.17N), and Tromso (69.65N) —
and generates 24 real-data geospatial tasks (6 tasks x 4 cities) compatible with `eval_test.py`.

Usage:
    python -m benchmarks.gsb_v2.build_osm_slice --out data/osm_eval
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def _aeqd(lat: float, lon: float) -> CRS:
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

CITIES = [
    # (slug, display_name, center_lat, center_lon, half_size_m)
    ("singapore", "Singapore (1.29N)", 1.2880, 103.8515, 320.0),
    ("xuzhou", "Xuzhou (34.26N)", 34.2610, 117.1850, 420.0),
    ("helsinki", "Helsinki (60.17N)", 60.1695, 24.9410, 320.0),
    ("tromso", "Tromso (69.65N)", 69.6492, 18.9553, 380.0),
]


def _bbox_for_city(lat: float, lon: float, half_m: float) -> tuple[float, float, float, float]:
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.15))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def fetch_osm_xml(bbox: tuple[float, float, float, float], timeout: float = 30.0) -> bytes:
    minlon, minlat, maxlon, maxlat = bbox
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={minlon:.6f},{minlat:.6f},{maxlon:.6f},{maxlat:.6f}"
    req = urllib.request.Request(url, headers={"User-Agent": "meta-rsi-research/0.4 (academic benchmark)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_osm_xml(xml_bytes: bytes) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    root = ET.fromstring(xml_bytes)
    nodes: dict[str, tuple[float, float]] = {}
    poi_rows = []

    for elem in root:
        if elem.tag == "node":
            nid = elem.attrib["id"]
            lon = float(elem.attrib["lon"])
            lat = float(elem.attrib["lat"])
            nodes[nid] = (lon, lat)
            tags = {c.attrib["k"]: c.attrib["v"] for c in elem if c.tag == "tag"}
            for key in ("amenity", "shop", "tourism", "public_transport", "highway"):
                if key in tags and tags[key]:
                    if key == "highway" and tags[key] not in ("bus_stop", "crossing"):
                        continue
                    poi_rows.append({"osm_id": int(nid), "category": str(tags[key]), "geometry": Point(lon, lat)})
                    break

    bldg_rows = []
    road_rows = []
    for elem in root:
        if elem.tag != "way":
            continue
        wid = int(elem.attrib["id"])
        nd_refs = [c.attrib["ref"] for c in elem if c.tag == "nd"]
        tags = {c.attrib["k"]: c.attrib["v"] for c in elem if c.tag == "tag"}
        coords = [nodes[r] for r in nd_refs if r in nodes]
        if len(coords) < 2:
            continue
        if "building" in tags and len(coords) >= 4 and coords[0] == coords[-1]:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.geom_type in ("Polygon", "MultiPolygon"):
                bldg_rows.append({"osm_id": wid, "btype": str(tags["building"]), "geometry": poly})
        elif "highway" in tags and len(coords) >= 2:
            line = LineString(coords)
            if not line.is_empty and line.length > 0:
                road_rows.append({"osm_id": wid, "highway_class": str(tags["highway"]), "geometry": line})

    g_bldg = gpd.GeoDataFrame(bldg_rows, crs="EPSG:4326") if bldg_rows else gpd.GeoDataFrame(columns=["osm_id", "btype", "geometry"], crs="EPSG:4326")
    g_road = gpd.GeoDataFrame(road_rows, crs="EPSG:4326") if road_rows else gpd.GeoDataFrame(columns=["osm_id", "highway_class", "geometry"], crs="EPSG:4326")
    g_poi = gpd.GeoDataFrame(poi_rows, crs="EPSG:4326") if poi_rows else gpd.GeoDataFrame(columns=["osm_id", "category", "geometry"], crs="EPSG:4326")
    return g_bldg, g_road, g_poi


def _build_city_tasks(
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

    # Exact AEQD ground truths on real OSM geometries
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

    # Also compute Web Mercator distortion check for audit logging
    b_wm = g_bldg.to_crs("EPSG:3857")
    wm_bldg_ha = float(b_wm.geometry.area.sum()) / 1e4
    wm_ratio = wm_bldg_ha / max(tot_bldg_ha, 1e-9)

    task_specs = [
        (
            "osm_building_area_ha",
            "float",
            tot_bldg_ha,
            ["buildings.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`). "
            "Write a Python script that computes the total footprint area of all buildings in hectares (ha). Print only the final number.",
        ),
        (
            "osm_road_length_km",
            "float",
            tot_road_km,
            ["roads.gpkg"],
            "The current directory contains `roads.gpkg` (line road segments with columns `osm_id`, `highway_class`). "
            "Write a Python script that computes the total length of all road segments in kilometres (km). Print only the final number.",
        ),
        (
            "osm_poi_buffer_building_area",
            "float",
            bldg_in_buf_ha,
            ["buildings.gpkg", "pois.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`) and "
            "`pois.gpkg` (point features with columns `osm_id`, `category`). "
            "Write a Python script that creates a 150-metre buffer around every POI, takes the union of all buffers, "
            "intersects each building footprint with that union, and computes the total intersected building area in hectares (ha). "
            "Print only the final number.",
        ),
        (
            "osm_large_building_area_ha",
            "float",
            large_bldg_ha,
            ["buildings.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`). "
            "Write a Python script that selects all buildings whose footprint area is strictly greater than 300 square metres "
            "and computes their combined area in hectares (ha). Print only the final number.",
        ),
        (
            "osm_nearest_poi_median_dist",
            "float",
            median_dist_m,
            ["buildings.gpkg", "pois.gpkg"],
            "The current directory contains `buildings.gpkg` (polygon building footprints with columns `osm_id`, `btype`) and "
            "`pois.gpkg` (point features with columns `osm_id`, `category`). "
            "Write a Python script that computes the Euclidean distance (in metres) from each building's centroid to its nearest POI, "
            "and prints the median distance across all buildings. Print only the final number.",
        ),
        (
            "osm_road_buffer_poi_count",
            "count",
            float(poi_near_road_cnt),
            ["roads.gpkg", "pois.gpkg"],
            "The current directory contains `roads.gpkg` (line road segments with columns `osm_id`, `highway_class`) and "
            "`pois.gpkg` (point features with columns `osm_id`, `category`). "
            "Write a Python script that counts how many POIs lie within 30 metres of at least one road segment. "
            "Print only the final integer.",
        ),
    ]

    layers_map = {"buildings.gpkg": g_bldg, "roads.gpkg": g_road, "pois.gpkg": g_poi}
    tasks = []
    for fam, ans_type, truth, need_files, prompt in task_specs:
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
            "city": slug,
            "group": slug,
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
    ap = argparse.ArgumentParser(description="Build real OSM 4-latitude urban GIS transfer test slice")
    ap.add_argument("--out", default="data/osm_eval")
    a = ap.parse_args()

    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    all_tasks: list[dict] = []
    for slug, disp, lat, lon, half_m in CITIES:
        bbox = _bbox_for_city(lat, lon, half_m)
        print(f"Fetching real OSM extract for {disp} bbox={[round(x, 4) for x in bbox]}...", flush=True)
        xml_bytes = fetch_osm_xml(bbox)
        g_bldg, g_road, g_poi = parse_osm_xml(xml_bytes)
        print(f"  parsed buildings={len(g_bldg)}, roads={len(g_road)}, pois={len(g_poi)}", flush=True)
        if len(g_bldg) < 10 or len(g_road) < 10 or len(g_poi) < 5:
            raise RuntimeError(f"Insufficient OSM features in {slug}: bldg={len(g_bldg)}, road={len(g_road)}, poi={len(g_poi)}")
        city_tasks = _build_city_tasks(root, slug, lat, lon, g_bldg, g_road, g_poi)
        print(f"  built {len(city_tasks)} tasks | Web Mercator area distortion = {city_tasks[0]['wm_area_ratio']:.3f}x", flush=True)
        all_tasks.extend(city_tasks)
        time.sleep(1.0)

    with open(root / "tasks.jsonl", "w") as fh:
        for t in all_tasks:
            fh.write(json.dumps(t) + "\n")
    print(f"Wrote {len(all_tasks)} real OSM tasks to {root}")


if __name__ == "__main__":
    main()

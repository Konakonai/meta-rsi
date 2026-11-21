"""GSB v2 task families: 20 families (15 seen + 5 held-out) across 4 modality groups.

Groups:
  A. Multi-step composite vector pipelines (6 seen + 2 held-out, low/high/polar/antimeridian)
  B. Raster/DEM + vector hybrid workflows (5 seen + 2 held-out, low/high/polar, GeoTIFF + nodata)
  C. Large-extent ellipsoidal geodesic tasks (2 seen + 1 held-out, low/mid/zone_edge)
  D. Pure non-projection methodological controls (2 seen, low/high/polar/antimeridian)

All local vector geometries keep a narrow clearance around x = 0 so that placing a scene at
`lon = 180.0` (antimeridian) yields 100% valid WGS84 individual polygons/lines on both sides of
+/-180 deg while queries cross the dateline and naive `total_bounds` wraps to [-179.99, +179.99].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import shapely
from pyproj import CRS, Geod
from rasterio.warp import Resampling
from shapely.geometry import LineString, Point, Polygon

from .geo import reproject_raster_to_crs

UNITS = {
    "m": (1.0, "metres"),
    "km": (1e-3, "kilometres"),
    "m2": (1.0, "square metres"),
    "km2": (1e-6, "square kilometres"),
    "ha": (1e-4, "hectares"),
}

LANDUSE = ["forest", "farmland", "urban", "wetland", "grassland"]
ROAD_TYPES = ["primary", "secondary", "residential", "arterial"]
SITE_TYPES = ["clinic", "shelter", "depot", "station"]
_GEOD = Geod(ellps="WGS84")
_SEAM_GAP = 140.0  # metres clearance on either side of x=0 for clean antimeridian splits


@dataclass
class RasterSpec:
    """Analytic physical raster field in local metres (origin = scene center)."""

    extent_m: float
    pixel_m: float
    field_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    valid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    force_4326_high: bool = False
    description: str = "elevation (metres)"


@dataclass
class Spec:
    params: dict
    layers: dict[str, gpd.GeoDataFrame]  # stem -> GeoDataFrame in local metres (or deg if large)
    kinds: dict[str, str]  # stem -> "point" | "line" | "polygon"
    columns: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    rasters: dict[str, RasterSpec] = field(default_factory=dict)  # stem -> RasterSpec


@dataclass
class Family:
    name: str
    group: str  # "composite_vector" | "raster_hybrid" | "large_extent" | "non_proj_control"
    answer: str  # "real" | "count"
    build: Callable
    question: Callable
    solve: Callable  # (layers, rasters, proj_fn, crs_obj, params) -> float
    held_out: bool = False
    large: bool = False
    has_raster: bool = False


# ---------- antimeridian-safe local geometry generators ----------

def _push_off_seam(x: float, gap: float = _SEAM_GAP) -> float:
    if abs(x) < gap:
        return gap if x >= 0 else -gap
    return x


def rand_points_split(rng: np.random.Generator, n: int, extent: float, n_clusters: int = 4) -> list[Point]:
    """Points distributed across both x < -gap and x > +gap so scenes straddle x = 0 cleanly."""
    half = n // 2
    pts = []
    for side, cnt in [(-1.0, half), (1.0, n - half)]:
        c_x = side * rng.uniform(_SEAM_GAP + 300.0, extent * 0.75, size=max(1, n_clusters // 2))
        c_y = rng.uniform(-extent * 0.65, extent * 0.65, size=len(c_x))
        which = rng.integers(0, len(c_x), cnt)
        sx = rng.normal(0, extent * 0.18, cnt)
        sy = rng.normal(0, extent * 0.22, cnt)
        xs = np.clip(c_x[which] + sx, -extent, extent)
        ys = np.clip(c_y[which] + sy, -extent, extent)
        for x, y in zip(xs, ys):
            x = side * max(abs(x), _SEAM_GAP + 20.0) if x * side < _SEAM_GAP + 20.0 else x
            pts.append(Point(float(np.clip(x, -extent, extent)), float(y)))
    rng.shuffle(pts)
    return pts


def rand_polygons_split(rng: np.random.Generator, n_cols_even: int, n_rows: int, extent: float) -> list[Polygon]:
    """Non-overlapping irregular polygons with an even column count leaving a clean seam at x = 0."""
    assert n_cols_even % 2 == 0
    half_cols = n_cols_even // 2
    cell_x = (extent - _SEAM_GAP) / half_cols
    cell_y = (2.0 * extent) / n_rows
    polys: list[Polygon] = []
    for side in (-1.0, 1.0):
        for i in range(half_cols):
            x_min = side * (_SEAM_GAP + i * cell_x)
            x_max = side * (_SEAM_GAP + (i + 1) * cell_x)
            lo_x, hi_x = min(x_min, x_max), max(x_min, x_max)
            for j in range(n_rows):
                lo_y = -extent + j * cell_y
                hi_y = lo_y + cell_y
                cx = 0.5 * (lo_x + hi_x) + rng.uniform(-0.06, 0.06) * cell_x
                cy = 0.5 * (lo_y + hi_y) + rng.uniform(-0.06, 0.06) * cell_y
                rx = rng.uniform(0.32, 0.44) * cell_x
                ry = rng.uniform(0.32, 0.44) * cell_y
                k = int(rng.integers(5, 8))
                ang = np.sort(rng.uniform(0, 2 * np.pi, k))
                rad_x = rx * rng.uniform(0.75, 1.0, k)
                rad_y = ry * rng.uniform(0.75, 1.0, k)
                xs = np.clip(cx + rad_x * np.cos(ang), lo_x + 15.0, hi_x - 15.0)
                ys = np.clip(cy + rad_y * np.sin(ang), lo_y + 15.0, hi_y - 15.0)
                poly = Polygon(zip(xs, ys))
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.area > 100.0:
                    polys.append(poly)
    return polys


def rand_lines_split(rng: np.random.Generator, n: int, extent: float) -> list[LineString]:
    """Polylines staying strictly on either x < -gap or x > +gap (never wrapping +/-180)."""
    lines = []
    for idx in range(n):
        side = -1.0 if idx % 2 == 0 else 1.0
        x = side * rng.uniform(_SEAM_GAP + 200.0, extent * 0.85)
        y = rng.uniform(-extent * 0.8, extent * 0.8)
        ang = rng.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _ in range(int(rng.integers(4, 8))):
            ang += rng.uniform(-0.5, 0.5)
            step = rng.uniform(extent * 0.08, extent * 0.16)
            nx = x + step * np.cos(ang)
            ny = np.clip(y + step * np.sin(ang), -extent * 0.92, extent * 0.92)
            # Keep strictly on the assigned side of the seam
            if side < 0:
                nx = np.clip(nx, -extent * 0.92, -_SEAM_GAP - 50.0)
            else:
                nx = np.clip(nx, _SEAM_GAP + 50.0, extent * 0.92)
            x, y = float(nx), float(ny)
            pts.append((x, y))
        lines.append(LineString(pts))
    return lines


def make_smooth_field(rng: np.random.Generator, base: float, amp: float, extent: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Deterministic smooth 2D field in local metres (elevation or environmental index)."""
    kx1 = rng.uniform(1.1, 2.2) * np.pi / extent
    ky1 = rng.uniform(1.1, 2.2) * np.pi / extent
    ph1 = rng.uniform(0, 2 * np.pi)
    kx2 = rng.uniform(1.8, 3.2) * np.pi / extent
    ky2 = rng.uniform(1.8, 3.2) * np.pi / extent
    ph2 = rng.uniform(0, 2 * np.pi)
    gx = rng.uniform(-0.35, 0.35) * amp / extent
    gy = rng.uniform(-0.35, 0.35) * amp / extent

    def _fn(xm: np.ndarray, ym: np.ndarray) -> np.ndarray:
        z = (
            base
            + 0.55 * amp * np.sin(kx1 * xm + ky1 * ym + ph1)
            + 0.35 * amp * np.cos(kx2 * xm - ky2 * ym + ph2)
            + gx * xm
            + gy * ym
        )
        return z

    return _fn


def make_nodata_mask(rng: np.random.Generator, extent: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Mask out a deterministic ~12% interior cloud/nodata patch so queried zones genuinely contain nodata pixels."""
    cx = float(rng.uniform(-0.22, 0.22) * extent)
    cy = float(rng.uniform(-0.22, 0.22) * extent)
    half_w = float(rng.uniform(0.24, 0.34) * extent)
    half_h = float(rng.uniform(0.24, 0.34) * extent)

    def _valid(xm: np.ndarray, ym: np.ndarray) -> np.ndarray:
        hole = (np.abs(xm - cx) <= half_w) & (np.abs(ym - cy) <= half_h)
        return ~hole

    return _valid


def _union_all(geoms):
    return shapely.union_all(list(geoms))


def _raster_metric_grid(raster_path: Path, target_crs: CRS, pixel_m: float = 50.0):
    return reproject_raster_to_crs(raster_path, target_crs, resampling=Resampling.bilinear, target_res_m=pixel_m)


def _rasterize_geom_mask(geom, shape: tuple[int, int], transform) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.geometry_mask([geom], out_shape=shape, transform=transform, invert=True)


# =====================================================================
# GROUP A: Multi-Step Composite Vector Pipelines (6 seen + 2 held-out)
# =====================================================================

def _build_buffer_exclusion_area(rng: np.random.Generator) -> Spec:
    extent = 7500.0
    # Place hubs & hazards so buffers overlap moderately across the seam
    hubs = rand_points_split(rng, int(rng.integers(8, 13)), extent * 0.72, n_clusters=4)
    hazards = rand_points_split(rng, int(rng.integers(4, 8)), extent * 0.72, n_clusters=4)
    r_in = int(rng.choice([1200, 1500, 1800]))
    r_ex = int(rng.choice([500, 700, 900]))
    unit = str(rng.choice(["km2", "ha"]))
    g_hubs = gpd.GeoDataFrame({"hub_id": np.arange(len(hubs))}, geometry=hubs)
    g_haz = gpd.GeoDataFrame({"hazard_id": np.arange(len(hazards))}, geometry=hazards)
    return Spec(
        params={"r_in": r_in, "r_ex": r_ex, "unit": unit},
        layers={"hubs": g_hubs, "hazards": g_haz},
        kinds={"hubs": "point", "hazards": "point"},
        columns={"hubs": [("hub_id", "int")], "hazards": [("hazard_id", "int")]},
    )


def _solve_buffer_exclusion_area(layers, rasters, proj, crs_obj, params) -> float:
    hubs = proj(layers["hubs"])
    haz = proj(layers["hazards"])
    inc = _union_all(hubs.geometry.buffer(params["r_in"]))
    exc = _union_all(haz.geometry.buffer(params["r_ex"]))
    net = inc.difference(exc)
    return float(net.area * UNITS[params["unit"]][0])


def _build_covered_pop_dedup(rng: np.random.Generator) -> Spec:
    extent = 7500.0
    blocks = rand_polygons_split(rng, 4, 4, extent)
    pops = rng.integers(200, 4000, len(blocks))
    # Place clinics near block centroids so multiple clinics overlap blocks
    clinics = rand_points_split(rng, int(rng.integers(8, 13)), extent * 0.75, n_clusters=4)
    radius = int(rng.choice([1800, 2200, 2600]))
    g_blk = gpd.GeoDataFrame({"block_id": np.arange(len(blocks)), "population": pops}, geometry=blocks)
    g_cl = gpd.GeoDataFrame({"site_id": np.arange(len(clinics))}, geometry=clinics)
    return Spec(
        params={"radius": radius},
        layers={"blocks": g_blk, "clinics": g_cl},
        kinds={"blocks": "polygon", "clinics": "point"},
        columns={"blocks": [("block_id", "int"), ("population", "int")], "clinics": [("site_id", "int")]},
    )


def _solve_covered_pop_dedup(layers, rasters, proj, crs_obj, params) -> float:
    blk = proj(layers["blocks"]).copy()
    cl = proj(layers["clinics"])
    cov = _union_all(cl.geometry.buffer(params["radius"]))
    cent = blk.geometry.centroid
    mask = cent.within(cov)
    return float(blk.loc[mask, "population"].sum())


def _build_area_weighted_pop_filter(rng: np.random.Generator) -> Spec:
    extent = 7500.0
    districts = rand_polygons_split(rng, 4, 4, extent)
    pops = rng.integers(1000, 9000, len(districts))
    sites = rand_points_split(rng, int(rng.integers(10, 16)), extent * 0.60, n_clusters=4)
    stypes = rng.choice(SITE_TYPES[:3], len(sites))
    target = str(rng.choice(SITE_TYPES[:3]))
    # Ensure at least 3 target sites
    for i in range(3):
        stypes[i] = target
    radius = int(rng.choice([2200, 2600, 3000]))
    g_dist = gpd.GeoDataFrame({"district_id": np.arange(len(districts)), "population": pops}, geometry=districts)
    g_sites = gpd.GeoDataFrame({"site_id": np.arange(len(sites)), "site_type": stypes}, geometry=sites)
    return Spec(
        params={"site_type": target, "radius": radius},
        layers={"districts": g_dist, "sites": g_sites},
        kinds={"districts": "polygon", "sites": "point"},
        columns={
            "districts": [("district_id", "int"), ("population", "int")],
            "sites": [("site_id", "int"), ("site_type", "str")],
        },
    )


def _solve_area_weighted_pop_filter(layers, rasters, proj, crs_obj, params) -> float:
    dist = proj(layers["districts"]).copy()
    sites = proj(layers["sites"])
    sel = sites[sites["site_type"] == params["site_type"]]
    buf = _union_all(sel.geometry.buffer(params["radius"]))
    total = 0.0
    for geom, pop in zip(dist.geometry, dist["population"]):
        a = geom.area
        if a > 0:
            total += float(pop) * (geom.intersection(buf).area / a)
    return total


def _build_multi_ring_density(rng: np.random.Generator) -> Spec:
    extent = 8000.0
    anchors = rand_points_split(rng, int(rng.integers(4, 7)), extent * 0.65, n_clusters=4)
    sensors = rand_points_split(rng, int(rng.integers(70, 110)), extent * 0.9, n_clusters=6)
    r_inner = int(rng.choice([800, 1000]))
    r_outer = int(rng.choice([2400, 2800, 3200]))
    g_anc = gpd.GeoDataFrame({"anchor_id": np.arange(len(anchors))}, geometry=anchors)
    g_sen = gpd.GeoDataFrame({"sensor_id": np.arange(len(sensors))}, geometry=sensors)
    return Spec(
        params={"r_inner": r_inner, "r_outer": r_outer},
        layers={"anchors": g_anc, "sensors": g_sen},
        kinds={"anchors": "point", "sensors": "point"},
        columns={"anchors": [("anchor_id", "int")], "sensors": [("sensor_id", "int")]},
    )


def _solve_multi_ring_density(layers, rasters, proj, crs_obj, params) -> float:
    anc = proj(layers["anchors"])
    sen = proj(layers["sensors"])
    outer = _union_all(anc.geometry.buffer(params["r_outer"]))
    inner = _union_all(anc.geometry.buffer(params["r_inner"]))
    ring = outer.difference(inner)
    area_km2 = ring.area * 1e-6
    cnt = int(sen.geometry.within(ring).sum())
    return float(cnt / area_km2) if area_km2 > 0 else 0.0


def _build_clipped_road_density(rng: np.random.Generator) -> Spec:
    extent = 7500.0
    zones = rand_polygons_split(rng, 4, 4, extent)
    classes = list(rng.choice(LANDUSE[:3], len(zones)))
    target = str(rng.choice(LANDUSE[:3]))
    for i in range(4):
        classes[i] = target
    roads = rand_lines_split(rng, int(rng.integers(24, 36)), extent)
    rtypes = rng.choice(ROAD_TYPES[:2], len(roads))
    g_zones = gpd.GeoDataFrame({"zone_id": np.arange(len(zones)), "landuse": classes}, geometry=zones)
    g_roads = gpd.GeoDataFrame({"road_id": np.arange(len(roads)), "road_type": rtypes}, geometry=roads)
    return Spec(
        params={"landuse": target},
        layers={"zones": g_zones, "roads": g_roads},
        kinds={"zones": "polygon", "roads": "line"},
        columns={
            "zones": [("zone_id", "int"), ("landuse", "str")],
            "roads": [("road_id", "int"), ("road_type", "str")],
        },
    )


def _solve_clipped_road_density(layers, rasters, proj, crs_obj, params) -> float:
    zones = proj(layers["zones"])
    roads = proj(layers["roads"])
    sel = zones[zones["landuse"] == params["landuse"]]
    u = _union_all(sel.geometry)
    area_km2 = u.area * 1e-6
    clipped_km = sum(g.intersection(u).length for g in roads.geometry) * 1e-3
    return float(clipped_km / area_km2) if area_km2 > 0 else 0.0


def _build_nearest_service_gap(rng: np.random.Generator) -> Spec:
    extent = 8000.0
    blocks = rand_polygons_split(rng, 4, 4, extent)
    pops = rng.integers(100, 2000, len(blocks))
    min_pop = int(np.median(pops))
    facs = rand_points_split(rng, int(rng.integers(10, 16)), extent * 0.85, n_clusters=4)
    status = list(rng.choice(["active", "closed"], len(facs)))
    for i in range(4):
        status[i] = "active"
    g_blk = gpd.GeoDataFrame({"block_id": np.arange(len(blocks)), "population": pops}, geometry=blocks)
    g_fac = gpd.GeoDataFrame({"facility_id": np.arange(len(facs)), "status": status}, geometry=facs)
    return Spec(
        params={"min_pop": min_pop},
        layers={"blocks": g_blk, "facilities": g_fac},
        kinds={"blocks": "polygon", "facilities": "point"},
        columns={
            "blocks": [("block_id", "int"), ("population", "int")],
            "facilities": [("facility_id", "int"), ("status", "str")],
        },
    )


def _solve_nearest_service_gap(layers, rasters, proj, crs_obj, params) -> float:
    blk = proj(layers["blocks"])
    fac = proj(layers["facilities"])
    sel_blk = blk[blk["population"] >= params["min_pop"]]
    act_fac = fac[fac["status"] == "active"]
    fac_union = _union_all(act_fac.geometry)
    dists = [c.distance(fac_union) for c in sel_blk.geometry.centroid]
    return float(max(dists) * 1e-3) if dists else 0.0


def _build_dual_buffer_overlap_pop(rng: np.random.Generator) -> Spec:
    extent = 7500.0
    blocks = rand_polygons_split(rng, 4, 4, extent)
    pops = rng.integers(300, 4500, len(blocks))
    stops = rand_points_split(rng, int(rng.integers(8, 12)), extent * 0.78, n_clusters=4)
    schools = rand_points_split(rng, int(rng.integers(8, 12)), extent * 0.78, n_clusters=4)
    r1 = int(rng.choice([2200, 2600, 3000]))
    r2 = int(rng.choice([2200, 2600, 3000]))
    g_blk = gpd.GeoDataFrame({"block_id": np.arange(len(blocks)), "population": pops}, geometry=blocks)
    g_st = gpd.GeoDataFrame({"stop_id": np.arange(len(stops))}, geometry=stops)
    g_sc = gpd.GeoDataFrame({"school_id": np.arange(len(schools))}, geometry=schools)
    return Spec(
        params={"r1": r1, "r2": r2},
        layers={"blocks": g_blk, "stops": g_st, "schools": g_sc},
        kinds={"blocks": "polygon", "stops": "point", "schools": "point"},
        columns={
            "blocks": [("block_id", "int"), ("population", "int")],
            "stops": [("stop_id", "int")],
            "schools": [("school_id", "int")],
        },
    )


def _solve_dual_buffer_overlap_pop(layers, rasters, proj, crs_obj, params) -> float:
    blk = proj(layers["blocks"])
    st = proj(layers["stops"])
    sc = proj(layers["schools"])
    u1 = _union_all(st.geometry.buffer(params["r1"]))
    u2 = _union_all(sc.geometry.buffer(params["r2"]))
    both = u1.intersection(u2)
    mask = blk.geometry.centroid.within(both)
    return float(blk.loc[mask, "population"].sum())


def _build_weighted_accessibility_ratio(rng: np.random.Generator) -> Spec:
    extent = 8000.0
    blocks = rand_polygons_split(rng, 4, 4, extent)
    pops = rng.integers(200, 5000, len(blocks))
    clinics = rand_points_split(rng, int(rng.integers(6, 10)), extent * 0.8, n_clusters=4)
    g_blk = gpd.GeoDataFrame({"block_id": np.arange(len(blocks)), "population": pops}, geometry=blocks)
    g_cl = gpd.GeoDataFrame({"clinic_id": np.arange(len(clinics))}, geometry=clinics)
    return Spec(
        params={},
        layers={"blocks": g_blk, "clinics": g_cl},
        kinds={"blocks": "polygon", "clinics": "point"},
        columns={"blocks": [("block_id", "int"), ("population", "int")], "clinics": [("clinic_id", "int")]},
    )


def _solve_weighted_accessibility_ratio(layers, rasters, proj, crs_obj, params) -> float:
    blk = proj(layers["blocks"])
    cl = proj(layers["clinics"])
    u = _union_all(cl.geometry)
    dists_km = np.array([c.distance(u) * 1e-3 for c in blk.geometry.centroid], dtype=float)
    pops = blk["population"].to_numpy(dtype=float)
    return float(np.sum(dists_km * pops) / np.sum(pops))


# =====================================================================
# GROUP B: Raster / DEM + Vector Hybrid Workflows (5 seen + 2 held-out)
# =====================================================================

def _build_raster_zonal_mean(rng: np.random.Generator) -> Spec:
    extent = 4500.0
    zones = rand_polygons_split(rng, 4, 4, extent * 0.92)
    classes = list(rng.choice(LANDUSE[:3], len(zones)))
    target = str(rng.choice(LANDUSE[:3]))
    for i in range(4):
        classes[i] = target
    g_zones = gpd.GeoDataFrame({"zone_id": np.arange(len(zones)), "landuse": classes}, geometry=zones)
    field_fn = make_smooth_field(rng, base=float(rng.uniform(350.0, 650.0)), amp=float(rng.uniform(120.0, 220.0)), extent=extent)
    valid_fn = make_nodata_mask(rng, extent)
    return Spec(
        params={"landuse": target},
        layers={"zones": g_zones},
        kinds={"zones": "polygon"},
        columns={"zones": [("zone_id", "int"), ("landuse", "str")]},
        rasters={"dem": RasterSpec(extent_m=extent, pixel_m=50.0, field_fn=field_fn, valid_fn=valid_fn, force_4326_high=True, description="elevation in metres")},
    )


def _solve_raster_zonal_mean(layers, rasters, proj, crs_obj, params) -> float:
    zones = proj(layers["zones"])
    sel = zones[zones["landuse"] == params["landuse"]]
    u = _union_all(sel.geometry)
    arr, tr, nodata = _raster_metric_grid(rasters["dem"], crs_obj, pixel_m=50.0)
    mask = _rasterize_geom_mask(u, arr.shape, tr) & np.isfinite(arr) & (~np.isclose(arr, nodata))
    vals = arr[mask]
    return float(np.mean(vals)) if len(vals) else 0.0


def _build_raster_threshold_area(rng: np.random.Generator) -> Spec:
    extent = 4500.0
    zones = rand_polygons_split(rng, 4, 4, extent * 0.92)
    classes = list(rng.choice(LANDUSE[:2], len(zones)))
    target = str(rng.choice(LANDUSE[:2]))
    for i in range(5):
        classes[i] = target
    base = float(rng.uniform(400.0, 600.0))
    amp = float(rng.uniform(150.0, 240.0))
    thresh = round(base - float(rng.uniform(10.0, 45.0)), 1)
    unit = str(rng.choice(["km2", "ha"]))
    g_zones = gpd.GeoDataFrame({"zone_id": np.arange(len(zones)), "landuse": classes}, geometry=zones)
    field_fn = make_smooth_field(rng, base=base, amp=amp, extent=extent)
    valid_fn = make_nodata_mask(rng, extent)
    return Spec(
        params={"landuse": target, "threshold": thresh, "unit": unit},
        layers={"zones": g_zones},
        kinds={"zones": "polygon"},
        columns={"zones": [("zone_id", "int"), ("landuse", "str")]},
        rasters={"dem": RasterSpec(extent_m=extent, pixel_m=50.0, field_fn=field_fn, valid_fn=valid_fn, description="elevation in metres")},
    )


def _solve_raster_threshold_area(layers, rasters, proj, crs_obj, params) -> float:
    zones = proj(layers["zones"])
    sel = zones[zones["landuse"] == params["landuse"]]
    u = _union_all(sel.geometry)
    arr, tr, nodata = _raster_metric_grid(rasters["dem"], crs_obj, pixel_m=50.0)
    mask = _rasterize_geom_mask(u, arr.shape, tr) & np.isfinite(arr) & (~np.isclose(arr, nodata))
    pix_area = abs(tr.a * tr.e)
    n_hit = int(np.sum(mask & (arr >= params["threshold"])))
    return float(n_hit * pix_area * UNITS[params["unit"]][0])


def _build_raster_slope_exceed_area(rng: np.random.Generator) -> Spec:
    extent = 4500.0
    # Smooth hills + interior study-area polygon (avoids rotated outer-border NaN erosion at poles)
    base = float(rng.uniform(500.0, 800.0))
    amp = float(rng.uniform(550.0, 750.0))
    slope_deg = float(rng.choice([5.0, 7.0, 9.0]))
    kx1 = rng.uniform(0.8, 1.35) * np.pi / extent
    ky1 = rng.uniform(0.8, 1.35) * np.pi / extent
    ph1 = rng.uniform(0, 2 * np.pi)
    kx2 = rng.uniform(1.1, 1.65) * np.pi / extent
    ky2 = rng.uniform(1.1, 1.65) * np.pi / extent
    ph2 = rng.uniform(0, 2 * np.pi)

    def field_fn(xm: np.ndarray, ym: np.ndarray) -> np.ndarray:
        return base + 0.6 * amp * np.sin(kx1 * xm + ky1 * ym + ph1) + 0.35 * amp * np.cos(kx2 * xm - ky2 * ym + ph2)

    valid_fn = make_nodata_mask(rng, extent)
    r_poly = extent * 0.78
    ang = np.sort(rng.uniform(0, 2 * np.pi, 8))
    rad = r_poly * rng.uniform(0.85, 1.0, 8)
    study_poly = Polygon(zip(rad * np.cos(ang), rad * np.sin(ang))).buffer(0)
    g_area = gpd.GeoDataFrame({"basin_id": [0]}, geometry=[study_poly])
    return Spec(
        params={"slope_deg": slope_deg},
        layers={"basin": g_area},
        kinds={"basin": "polygon"},
        columns={"basin": [("basin_id", "int")]},
        rasters={"dem": RasterSpec(extent_m=extent, pixel_m=50.0, field_fn=field_fn, valid_fn=valid_fn, description="elevation in metres")},
    )


def _solve_raster_slope_exceed_area(layers, rasters, proj, crs_obj, params) -> float:
    basin = proj(layers["basin"])
    u = _union_all(basin.geometry)
    arr, tr, nodata = _raster_metric_grid(rasters["dem"], crs_obj, pixel_m=50.0)
    valid = np.isfinite(arr) & (~np.isclose(arr, nodata))
    work = arr.astype(np.float64).copy()
    work[~valid] = np.nan
    dx = abs(tr.a)
    dy = abs(tr.e)
    dz_dy, dz_dx = np.gradient(work, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    in_basin = _rasterize_geom_mask(u, arr.shape, tr)
    hit = in_basin & np.isfinite(slope) & (slope > params["slope_deg"])
    pix_area_ha = (dx * dy) * 1e-4
    return float(np.sum(hit) * pix_area_ha)


def _build_raster_buffer_extract(rng: np.random.Generator) -> Spec:
    extent = 4500.0
    sensors = rand_points_split(rng, int(rng.integers(6, 11)), extent * 0.7, n_clusters=4)
    status = list(rng.choice(["active", "offline"], len(sensors)))
    for i in range(3):
        status[i] = "active"
    radius = int(rng.choice([700, 900, 1100]))
    g_sen = gpd.GeoDataFrame({"sensor_id": np.arange(len(sensors)), "status": status}, geometry=sensors)
    field_fn = make_smooth_field(rng, base=float(rng.uniform(40.0, 70.0)), amp=float(rng.uniform(20.0, 35.0)), extent=extent)
    valid_fn = make_nodata_mask(rng, extent)
    return Spec(
        params={"radius": radius},
        layers={"sensors": g_sen},
        kinds={"sensors": "point"},
        columns={"sensors": [("sensor_id", "int"), ("status", "str")]},
        rasters={"index": RasterSpec(extent_m=extent, pixel_m=50.0, field_fn=field_fn, valid_fn=valid_fn, description="environmental index value")},
    )


def _solve_raster_buffer_extract(layers, rasters, proj, crs_obj, params) -> float:
    sen = proj(layers["sensors"])
    act = sen[sen["status"] == "active"]
    u = _union_all(act.geometry.buffer(params["radius"]))
    arr, tr, nodata = _raster_metric_grid(rasters["index"], crs_obj, pixel_m=50.0)
    mask = _rasterize_geom_mask(u, arr.shape, tr) & np.isfinite(arr) & (~np.isclose(arr, nodata))
    vals = arr[mask]
    return float(np.mean(vals)) if len(vals) else 0.0


def _build_raster_pop_exposure(rng: np.random.Generator) -> Spec:
    extent = 4500.0
    blocks = rand_polygons_split(rng, 4, 4, extent * 0.9)
    pops = rng.integers(300, 4000, len(blocks))
    base = float(rng.uniform(50.0, 80.0))
    amp = float(rng.uniform(25.0, 40.0))
    thresh = round(base + float(rng.uniform(-5.0, 8.0)), 1)
    g_blk = gpd.GeoDataFrame({"block_id": np.arange(len(blocks)), "population": pops}, geometry=blocks)
    field_fn = make_smooth_field(rng, base=base, amp=amp, extent=extent)
    valid_fn = make_nodata_mask(rng, extent)
    return Spec(
        params={"threshold": thresh},
        layers={"blocks": g_blk},
        kinds={"blocks": "polygon"},
        columns={"blocks": [("block_id", "int"), ("population", "int")]},
        rasters={"hazard_map": RasterSpec(extent_m=extent, pixel_m=50.0, field_fn=field_fn, valid_fn=valid_fn, force_4326_high=True, description="heat/pollution exposure index")},
    )


def _solve_raster_pop_exposure(layers, rasters, proj, crs_obj, params) -> float:
    blk = proj(layers["blocks"])
    arr, tr, nodata = _raster_metric_grid(rasters["hazard_map"], crs_obj, pixel_m=50.0)
    valid_all = np.isfinite(arr) & (~np.isclose(arr, nodata))
    total = 0.0
    for geom, pop in zip(blk.geometry, blk["population"]):
        m = _rasterize_geom_mask(geom, arr.shape, tr) & valid_all
        if np.any(m) and float(np.mean(arr[m])) >= params["threshold"]:
            total += float(pop)
    return total


def _build_raster_corridor_elevation(rng: np.random.Generator) -> Spec:
    extent = 4500.0
    roads = rand_lines_split(rng, int(rng.integers(10, 16)), extent * 0.88)
    rtypes = list(rng.choice(ROAD_TYPES[:2], len(roads)))
    for i in range(4):
        rtypes[i] = "primary"
    width = int(rng.choice([300, 400, 500]))
    g_rd = gpd.GeoDataFrame({"road_id": np.arange(len(roads)), "road_type": rtypes}, geometry=roads)
    field_fn = make_smooth_field(rng, base=float(rng.uniform(300.0, 600.0)), amp=float(rng.uniform(140.0, 240.0)), extent=extent)
    valid_fn = make_nodata_mask(rng, extent)
    return Spec(
        params={"width": width},
        layers={"roads": g_rd},
        kinds={"roads": "line"},
        columns={"roads": [("road_id", "int"), ("road_type", "str")]},
        rasters={"dem": RasterSpec(extent_m=extent, pixel_m=50.0, field_fn=field_fn, valid_fn=valid_fn, description="elevation in metres")},
    )


def _solve_raster_corridor_elevation(layers, rasters, proj, crs_obj, params) -> float:
    rd = proj(layers["roads"])
    sel = rd[rd["road_type"] == "primary"]
    u = _union_all(sel.geometry.buffer(params["width"]))
    arr, tr, nodata = _raster_metric_grid(rasters["dem"], crs_obj, pixel_m=50.0)
    mask = _rasterize_geom_mask(u, arr.shape, tr) & np.isfinite(arr) & (~np.isclose(arr, nodata))
    vals = arr[mask]
    return float(np.mean(vals)) if len(vals) else 0.0


def _build_raster_hotspot_in_buffer(rng: np.random.Generator) -> Spec:
    extent = 4500.0
    facs = rand_points_split(rng, int(rng.integers(5, 9)), extent * 0.72, n_clusters=4)
    radius = int(rng.choice([1200, 1500, 1800]))
    base = float(rng.uniform(60.0, 90.0))
    amp = float(rng.uniform(30.0, 45.0))
    thresh = round(base + float(rng.uniform(-5.0, 10.0)), 1)
    g_fac = gpd.GeoDataFrame({"facility_id": np.arange(len(facs))}, geometry=facs)
    field_fn = make_smooth_field(rng, base=base, amp=amp, extent=extent)
    valid_fn = make_nodata_mask(rng, extent)
    return Spec(
        params={"radius": radius, "threshold": thresh},
        layers={"facilities": g_fac},
        kinds={"facilities": "point"},
        columns={"facilities": [("facility_id", "int")]},
        rasters={"intensity": RasterSpec(extent_m=extent, pixel_m=50.0, field_fn=field_fn, valid_fn=valid_fn, description="hotspot intensity value")},
    )


def _solve_raster_hotspot_in_buffer(layers, rasters, proj, crs_obj, params) -> float:
    fac = proj(layers["facilities"])
    u = _union_all(fac.geometry.buffer(params["radius"]))
    arr, tr, nodata = _raster_metric_grid(rasters["intensity"], crs_obj, pixel_m=50.0)
    mask = _rasterize_geom_mask(u, arr.shape, tr) & np.isfinite(arr) & (~np.isclose(arr, nodata))
    pix_area_km2 = abs(tr.a * tr.e) * 1e-6
    return float(np.sum(mask & (arr >= params["threshold"])) * pix_area_km2)


# =====================================================================
# GROUP C: Large-Extent Ellipsoidal Geodesic (2 seen + 1 held-out)
# =====================================================================

def _build_large_region_area_v2(rng: np.random.Generator) -> Spec:
    # 4-6 large polygons spanning ~35-55 deg longitude & ~14-22 deg latitude
    n = int(rng.integers(4, 7))
    polys = []
    span_x = float(rng.uniform(36.0, 52.0))
    span_y = float(rng.uniform(14.0, 22.0))
    for i in range(n):
        cx = -0.5 * span_x + (i + 0.5) * (span_x / n) + rng.uniform(-1.0, 1.0)
        cy = rng.uniform(-0.35 * span_y, 0.35 * span_y)
        rx = rng.uniform(3.0, 4.8)
        ry = rng.uniform(2.5, 4.2)
        ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
        poly = Polygon(zip(cx + rx * np.cos(ang), cy + ry * np.sin(ang)))
        polys.append(poly.buffer(0))
    classes = list(rng.choice(["biome_a", "biome_b"], n))
    target = "biome_a"
    classes[0] = target
    classes[-1] = target
    g = gpd.GeoDataFrame({"region_id": np.arange(n), "biome": classes}, geometry=polys)
    return Spec(
        params={"biome": target},
        layers={"regions": g},
        kinds={"regions": "polygon"},
        columns={"regions": [("region_id", "int"), ("biome", "str")]},
    )


def _solve_large_region_area_v2(layers, rasters, proj, crs_obj, params) -> float:
    g = layers["regions"]
    sel = g[g["biome"] == params["biome"]]
    if proj is not None:
        return float(proj(sel).geometry.area.sum() * 1e-6)
    g4326 = sel.to_crs("EPSG:4326") if sel.crs else sel
    tot = 0.0
    for geom in g4326.geometry:
        a, _ = _GEOD.geometry_area_perimeter(geom)
        tot += abs(a)
    return float(tot * 1e-6)


def _build_large_route_length_v2(rng: np.random.Generator) -> Spec:
    n = int(rng.integers(3, 6))
    lines = []
    for _ in range(n):
        span = float(rng.uniform(48.0, 72.0))
        xs = np.linspace(-0.5 * span, 0.5 * span, int(rng.integers(6, 10)))
        ys = rng.uniform(-12.0, 12.0, len(xs))
        lines.append(LineString(zip(xs, ys)))
    corridors = list(rng.choice(["freight", "passenger"], n))
    corridors[0] = "freight"
    g = gpd.GeoDataFrame({"route_id": np.arange(n), "corridor": corridors}, geometry=lines)
    return Spec(
        params={"corridor": "freight"},
        layers={"routes": g},
        kinds={"routes": "line"},
        columns={"routes": [("route_id", "int"), ("corridor", "str")]},
    )


def _solve_large_route_length_v2(layers, rasters, proj, crs_obj, params) -> float:
    g = layers["routes"]
    sel = g[g["corridor"] == params["corridor"]]
    if proj is not None:
        return float(proj(sel).geometry.length.sum() * 1e-3)
    g4326 = sel.to_crs("EPSG:4326") if sel.crs else sel
    return float(sum(_GEOD.geometry_length(geom) for geom in g4326.geometry) * 1e-3)


def _build_large_max_geodesic_span(rng: np.random.Generator) -> Spec:
    n = int(rng.integers(10, 16))
    span_x = float(rng.uniform(46.0, 70.0))
    span_y = float(rng.uniform(16.0, 26.0))
    xs = rng.uniform(-0.5 * span_x, 0.5 * span_x, n)
    ys = rng.uniform(-0.5 * span_y, 0.5 * span_y, n)
    xs[0], xs[1] = -0.5 * span_x, 0.5 * span_x
    pts = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
    tiers = list(rng.choice(["major", "minor"], n))
    tiers[0], tiers[1] = "major", "major"
    g = gpd.GeoDataFrame({"city_id": np.arange(n), "tier": tiers}, geometry=pts)
    return Spec(
        params={"tier": "major"},
        layers={"cities": g},
        kinds={"cities": "point"},
        columns={"cities": [("city_id", "int"), ("tier", "str")]},
    )


def _solve_large_max_geodesic_span(layers, rasters, proj, crs_obj, params) -> float:
    g = layers["cities"]
    sel = g[g["tier"] == params["tier"]]
    if proj is not None:
        gp = proj(sel)
        coords = [(p.x, p.y) for p in gp.geometry]
        best = 0.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = float(np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]))
                if d > best:
                    best = d
        return best * 1e-3
    g4326 = sel.to_crs("EPSG:4326") if sel.crs else sel
    coords = [(p.x, p.y) for p in g4326.geometry]
    best = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            _, _, d = _GEOD.inv(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if d > best:
                best = d
    return float(best * 1e-3)


# =====================================================================
# GROUP D: Pure Non-Projection Controls (2 seen)
# =====================================================================

def _build_topological_containment_v2(rng: np.random.Generator) -> Spec:
    extent = 7500.0
    zones = rand_polygons_split(rng, 4, 4, extent)
    classes = list(rng.choice(LANDUSE[:3], len(zones)))
    target = str(rng.choice(LANDUSE[:3]))
    for i in range(4):
        classes[i] = target
    pts = rand_points_split(rng, int(rng.integers(50, 90)), extent * 0.95, n_clusters=6)
    g_z = gpd.GeoDataFrame({"zone_id": np.arange(len(zones)), "landuse": classes}, geometry=zones)
    g_p = gpd.GeoDataFrame({"sample_id": np.arange(len(pts))}, geometry=pts)
    return Spec(
        params={"landuse": target},
        layers={"zones": g_z, "samples": g_p},
        kinds={"zones": "polygon", "samples": "point"},
        columns={"zones": [("zone_id", "int"), ("landuse", "str")], "samples": [("sample_id", "int")]},
    )


def _solve_topological_containment_v2(layers, rasters, proj, crs_obj, params) -> float:
    z = proj(layers["zones"])
    p = proj(layers["samples"])
    sel = z[z["landuse"] == params["landuse"]]
    u = _union_all(sel.geometry)
    return float(int(p.geometry.within(u).sum()))


def _build_district_mean_zero_trap_v2(rng: np.random.Generator) -> Spec:
    extent = 7500.0
    dists = rand_polygons_split(rng, 4, 4, extent)
    # Sparse points so several districts have zero points!
    pts = rand_points_split(rng, int(rng.integers(10, 18)), extent * 0.85, n_clusters=3)
    g_d = gpd.GeoDataFrame({"district_id": np.arange(len(dists))}, geometry=dists)
    g_p = gpd.GeoDataFrame({"event_id": np.arange(len(pts))}, geometry=pts)
    return Spec(
        params={},
        layers={"districts": g_d, "events": g_p},
        kinds={"districts": "polygon", "events": "point"},
        columns={"districts": [("district_id", "int")], "events": [("event_id", "int")]},
    )


def _solve_district_mean_zero_trap_v2(layers, rasters, proj, crs_obj, params) -> float:
    d = proj(layers["districts"])
    p = proj(layers["events"])
    counts = [int(p.geometry.within(g).sum()) for g in d.geometry]
    return float(np.mean(counts)) if counts else 0.0


# =====================================================================
# Registry (20 families: 15 seen + 5 held-out)
# =====================================================================

FAMILIES: list[Family] = [
    # Group A — Multi-step composite vector (6 seen + 2 held-out)
    Family(
        "buffer_exclusion_area", "composite_vector", "real",
        _build_buffer_exclusion_area,
        lambda p: f"computes the total area in {UNITS[p['unit']][1]} that is within {p['r_in']} metres of at least one hub in `hubs` AND at least {p['r_ex']} metres away from every hazard in `hazards`",
        _solve_buffer_exclusion_area,
    ),
    Family(
        "covered_pop_dedup", "composite_vector", "real",
        _build_covered_pop_dedup,
        lambda p: f"computes the total `population` of unique residential blocks in `blocks` whose centroid is within {p['radius']} metres of at least one clinic in `clinics` (count each covered block once)",
        _solve_covered_pop_dedup,
    ),
    Family(
        "area_weighted_pop_filter", "composite_vector", "real",
        _build_area_weighted_pop_filter,
        lambda p: f"estimates the total population within {p['radius']} metres of any site with `site_type == '{p['site_type']}'` in `sites`, using area-weighted interpolation from overlapping polygons in `districts`",
        _solve_area_weighted_pop_filter,
    ),
    Family(
        "multi_ring_density", "composite_vector", "real",
        _build_multi_ring_density,
        lambda p: f"computes the density (sensors per square kilometre) of `sensors` located inside the ring zone that is within {p['r_outer']} metres of at least one anchor in `anchors` but farther than {p['r_inner']} metres from all anchors",
        _solve_multi_ring_density,
    ),
    Family(
        "clipped_road_density", "composite_vector", "real",
        _build_clipped_road_density,
        lambda p: f"computes the road density (kilometres of road per square kilometre of zone area) of `roads` clipped strictly inside `zones` polygons with `landuse == '{p['landuse']}'`",
        _solve_clipped_road_density,
    ),
    Family(
        "nearest_service_gap", "composite_vector", "real",
        _build_nearest_service_gap,
        lambda p: f"computes the maximum distance in kilometres from the centroid of any block in `blocks` with `population >= {p['min_pop']}` to its nearest facility with `status == 'active'` in `facilities`",
        _solve_nearest_service_gap,
    ),
    Family(
        "dual_buffer_overlap_pop", "composite_vector", "real",
        _build_dual_buffer_overlap_pop,
        lambda p: f"computes the total `population` of blocks in `blocks` whose centroid is simultaneously within {p['r1']} metres of at least one stop in `stops` AND within {p['r2']} metres of at least one school in `schools`",
        _solve_dual_buffer_overlap_pop,
        held_out=True,
    ),
    Family(
        "weighted_accessibility_ratio", "composite_vector", "real",
        _build_weighted_accessibility_ratio,
        lambda p: "computes the population-weighted mean distance in kilometres from each block centroid in `blocks` to its nearest clinic in `clinics`",
        _solve_weighted_accessibility_ratio,
        held_out=True,
    ),
    # Group B — Raster / DEM + vector hybrid (5 seen + 2 held-out)
    Family(
        "raster_zonal_mean", "raster_hybrid", "real",
        _build_raster_zonal_mean,
        lambda p: f"computes the mean valid elevation in `dem` across all valid (non-nodata) pixels falling inside `zones` polygons with `landuse == '{p['landuse']}'`",
        _solve_raster_zonal_mean,
        has_raster=True,
    ),
    Family(
        "raster_threshold_area", "raster_hybrid", "real",
        _build_raster_threshold_area,
        lambda p: f"computes the total ground area in {UNITS[p['unit']][1]} inside `zones` polygons with `landuse == '{p['landuse']}'` where valid (non-nodata) elevation in `dem` is at least {p['threshold']} metres",
        _solve_raster_threshold_area,
        has_raster=True,
    ),
    Family(
        "raster_slope_exceed_area", "raster_hybrid", "real",
        _build_raster_slope_exceed_area,
        lambda p: f"computes the total ground area in hectares inside `basin` of valid (non-nodata) terrain in `dem` where the terrain slope angle exceeds {p['slope_deg']} degrees",
        _solve_raster_slope_exceed_area,
        has_raster=True,
    ),
    Family(
        "raster_buffer_extract", "raster_hybrid", "real",
        _build_raster_buffer_extract,
        lambda p: f"computes the mean valid (non-nodata) pixel value in `index` within {p['radius']} metres of any sensor with `status == 'active'` in `sensors`",
        _solve_raster_buffer_extract,
        has_raster=True,
    ),
    Family(
        "raster_pop_exposure", "raster_hybrid", "real",
        _build_raster_pop_exposure,
        lambda p: f"computes the total `population` of blocks in `blocks` whose mean valid (non-nodata) value in `hazard_map` is at least {p['threshold']}",
        _solve_raster_pop_exposure,
        has_raster=True,
    ),
    Family(
        "raster_corridor_elevation", "raster_hybrid", "real",
        _build_raster_corridor_elevation,
        lambda p: f"computes the mean valid (non-nodata) elevation in `dem` within a {p['width']}-metre corridor around all roads with `road_type == 'primary'` in `roads`",
        _solve_raster_corridor_elevation,
        held_out=True,
        has_raster=True,
    ),
    Family(
        "raster_hotspot_in_buffer", "raster_hybrid", "real",
        _build_raster_hotspot_in_buffer,
        lambda p: f"computes the total ground area in square kilometres that is within {p['radius']} metres of any facility in `facilities` AND has a valid (non-nodata) pixel value >= {p['threshold']} in `intensity`",
        _solve_raster_hotspot_in_buffer,
        held_out=True,
        has_raster=True,
    ),
    # Group C — Large-extent ellipsoidal geodesic (2 seen + 1 held-out)
    Family(
        "large_region_area_v2", "large_extent", "real",
        _build_large_region_area_v2,
        lambda p: f"computes the total ellipsoidal ground area in square kilometres of all polygons in `regions` with `biome == '{p['biome']}'`",
        _solve_large_region_area_v2,
        large=True,
    ),
    Family(
        "large_route_length_v2", "large_extent", "real",
        _build_large_route_length_v2,
        lambda p: f"computes the total geodesic length in kilometres of all routes in `routes` with `corridor == '{p['corridor']}'`",
        _solve_large_route_length_v2,
        large=True,
    ),
    Family(
        "large_max_geodesic_span", "large_extent", "real",
        _build_large_max_geodesic_span,
        lambda p: f"computes the maximum geodesic distance in kilometres between any pair of cities in `cities` with `tier == '{p['tier']}'`",
        _solve_large_max_geodesic_span,
        held_out=True,
        large=True,
    ),
    # Group D — Pure non-projection methodological controls (2 seen)
    Family(
        "topological_containment_v2", "non_proj_control", "count",
        _build_topological_containment_v2,
        lambda p: f"counts how many points in `samples` fall strictly inside polygons with `landuse == '{p['landuse']}'` in `zones`",
        _solve_topological_containment_v2,
    ),
    Family(
        "district_mean_zero_trap_v2", "non_proj_control", "real",
        _build_district_mean_zero_trap_v2,
        lambda p: "computes the mean number of points in `events` per district across ALL polygons in `districts` (including districts containing zero events)",
        _solve_district_mean_zero_trap_v2,
    ),
]

BY_NAME: dict[str, Family] = {f.name: f for f in FAMILIES}

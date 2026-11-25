"""E1-fixed Polar LAEA probe: geodesic densification + antimeridian/pole splitting (RFC 7946 §3.1.9).

Identical to `runs/v8_s4_wave2_v2/certified_probes/probe_r2_polar_laea.py` except for two geometric
fixes applied to vector layers:
  1. Geographic geometries spanning > 0.5 deg are densified along WGS84 geodesics every 10 km
     before projecting into LAEA, preserving ellipsoidal area to < 0.01% across polar relocation.
  2. Any relocated geometry touching the ±180 deg meridian is split in the local LAEA tangent plane
     along a thin (5 m) slit through ±180 deg (and along the full 0/180 deg diameter + 25 m pole cap
     if a polygon encloses the pole or still spans > 180 deg) before inverse-projecting to WGS84, so
     no individual output ring spans > 180 deg of longitude or jumps across the antimeridian.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from pyproj import CRS, Transformer, Geod
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import transform as shapely_transform, unary_union
import shapely


def _find_coord_cols(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    for lon_k, lat_k in [("lon", "lat"), ("longitude", "latitude"), ("x", "y")]:
        if lon_k in cols_lower and lat_k in cols_lower:
            return cols_lower[lon_k], cols_lower[lat_k]
    return None, None


def _get_workspace_center(src_dir: Path):
    xs, ys = [], []
    to_wgs84_cache = {}

    def _to_wgs84(crs_obj):
        key = crs_obj.to_string()
        if key not in to_wgs84_cache:
            to_wgs84_cache[key] = Transformer.from_crs(crs_obj, CRS.from_epsg(4326), always_xy=True)
        return to_wgs84_cache[key]

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in {".geojson", ".gpkg"}:
            gdf = gpd.read_file(p)
            if not gdf.empty and gdf.crs is not None:
                minx, miny, maxx, maxy = gdf.total_bounds
                cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                lon, lat = _to_wgs84(CRS.from_user_input(gdf.crs)).transform(cx, cy)
                xs.append(lon)
                ys.append(lat)
        elif suf == ".csv":
            df = pd.read_csv(p)
            lon_c, lat_c = _find_coord_cols(df)
            if lon_c and not df.empty:
                xs.append(float(df[lon_c].mean()))
                ys.append(float(df[lat_c].mean()))
        elif suf == ".tif":
            with rasterio.open(p) as src:
                if src.crs is not None:
                    b = src.bounds
                    cx, cy = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
                    lon, lat = _to_wgs84(CRS.from_user_input(src.crs)).transform(cx, cy)
                    xs.append(lon)
                    ys.append(lat)

    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def _meridian_slits_local(wgs84: CRS, dst_local_crs: CRS) -> tuple[Polygon, Polygon]:
    """Build thin (5 m) local LAEA slits along ±180° (and full 0°/180° diameter through the pole).

    Cutting only along ±180° splits antimeridian crossings; when a polygon encloses the North Pole,
    cutting the full 0°/180° diameter splits it into Eastern [0, 180] and Western [-180, 0] halves so
    no individual ring can span > 180° of longitude while preserving geodesic area to < 0.01%.
    """
    t = Transformer.from_crs(wgs84, dst_local_crs, always_xy=True)
    lats = np.concatenate([np.linspace(0.0, 89.9, 1000), np.linspace(89.9, 90.0, 200)])

    def _meridian_line(lon_deg: float) -> LineString:
        xs, ys = t.transform(np.full_like(lats, lon_deg), lats)
        ok = np.isfinite(xs) & np.isfinite(ys)
        return LineString(np.column_stack([xs[ok], ys[ok]]))

    px, py = t.transform(0.0, 90.0)
    pole_cap = Point(px, py).buffer(25.0)
    am_slit = unary_union([_meridian_line(180.0).buffer(5.0, cap_style="flat"), pole_cap])
    full_slit = unary_union([am_slit, _meridian_line(0.0).buffer(5.0, cap_style="flat")])
    return am_slit, full_slit


def _densify_ring_geod(coords, geod: Geod, step_m: float = 10_000.0) -> list[tuple[float, float]]:
    pts = list(coords)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        _, _, dist = geod.inv(x1, y1, x2, y2)
        if np.isfinite(dist) and dist > step_m:
            n_mid = max(1, int(dist // step_m))
            out.extend(geod.npts(x1, y1, x2, y2, n_mid))
        out.append((x2, y2))
    return out


def _densify_geod_wgs84(geom, geod: Geod, step_m: float = 10_000.0):
    if geom is None or geom.is_empty:
        return geom
    gt = geom.geom_type
    if gt == "LineString":
        return LineString(_densify_ring_geod(geom.coords, geod, step_m))
    if gt == "Polygon":
        ext = _densify_ring_geod(geom.exterior.coords, geod, step_m)
        ints = [_densify_ring_geod(r.coords, geod, step_m) for r in geom.interiors]
        return Polygon(ext, ints)
    if gt == "MultiPolygon":
        return MultiPolygon([_densify_geod_wgs84(g, geod, step_m) for g in geom.geoms])
    if gt == "MultiLineString":
        return MultiLineString([_densify_geod_wgs84(g, geod, step_m) for g in geom.geoms])
    return geom


def _max_part_lon_span(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    if hasattr(geom, "geoms"):
        return max((_max_part_lon_span(g) for g in geom.geoms), default=0.0)
    b = geom.bounds
    return float(b[2] - b[0])


def _ring_crosses_am(coords) -> bool:
    arr = np.asarray(coords)
    if len(arr) < 2:
        return False
    return bool(np.any(np.abs(np.diff(arr[:, 0])) > 180.0))


def _geom_crosses_am(geom) -> bool:
    if geom is None or geom.is_empty:
        return False
    gt = geom.geom_type
    if gt == "Point" or gt == "MultiPoint":
        return False
    if gt == "LineString":
        return _ring_crosses_am(geom.coords)
    if gt == "Polygon":
        return _ring_crosses_am(geom.exterior.coords) or any(_ring_crosses_am(r.coords) for r in geom.interiors)
    if hasattr(geom, "geoms"):
        return any(_geom_crosses_am(g) for g in geom.geoms)
    return False


def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    lon0, lat0 = _get_workspace_center(src_dir)
    target_lat = 86.0
    target_lon = lon0

    src_local_crs = CRS.from_proj4(
        f"+proj=laea +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    dst_local_crs = CRS.from_proj4(
        f"+proj=laea +lat_0={target_lat} +lon_0={target_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    geod = Geod(ellps="WGS84")
    am_slit, full_slit = _meridian_slits_local(wgs84, dst_local_crs)

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        dst_p = dst_dir / p.name
        suf = p.suffix.lower()

        if suf in {".geojson", ".gpkg"}:
            gdf = gpd.read_file(p)
            if not gdf.empty and gdf.crs is not None:
                orig_crs = CRS.from_user_input(gdf.crs)
                t_to_local = Transformer.from_crs(orig_crs, src_local_crs, always_xy=True)
                t_from_local = Transformer.from_crs(dst_local_crs, wgs84, always_xy=True)

                def _relocate_geom(geom):
                    if geom is None or geom.is_empty:
                        return geom
                    # For large-extent geometries (> 0.5 deg), densify edges along WGS84 geodesics
                    # BEFORE projecting into LAEA so exact ellipsoidal area is preserved (< 0.01%).
                    g_src = geom
                    if orig_crs.is_geographic:
                        b = geom.bounds
                        if max(b[2] - b[0], b[3] - b[1]) > 0.5:
                            g_src = _densify_geod_wgs84(geom, geod, step_m=10_000.0)
                    g_local = shapely_transform(t_to_local.transform, g_src)
                    g_cut = g_local.difference(am_slit) if g_local.intersects(am_slit) else g_local
                    g_wgs = shapely_transform(t_from_local.transform, g_cut)
                    if _geom_crosses_am(g_wgs) or _max_part_lon_span(g_wgs) > 180.0:
                        g_cut2 = shapely.segmentize(g_local, max_segment_length=2500.0).difference(full_slit)
                        g_wgs = shapely_transform(t_from_local.transform, g_cut2)
                    return g_wgs

                gdf = gdf.copy()
                gdf["geometry"] = gdf["geometry"].apply(_relocate_geom)
                gdf = gdf.set_crs(wgs84, allow_override=True)

            driver = "GeoJSON" if suf == ".geojson" else "GPKG"
            gdf.to_file(dst_p, driver=driver)

        elif suf == ".csv":
            df = pd.read_csv(p)
            lon_c, lat_c = _find_coord_cols(df)
            if lon_c and not df.empty:
                t_to_local = Transformer.from_crs(wgs84, src_local_crs, always_xy=True)
                t_from_local = Transformer.from_crs(dst_local_crs, wgs84, always_xy=True)
                lx, ly = t_to_local.transform(df[lon_c].to_numpy(), df[lat_c].to_numpy())
                nlon, nlat = t_from_local.transform(lx, ly)
                df = df.copy()
                df[lon_c] = np.round(nlon, 7)
                df[lat_c] = np.round(nlat, 7)
            df.to_csv(dst_p, index=False)

        elif suf == ".tif":
            with rasterio.open(p) as src:
                data = src.read()
                profile = src.profile.copy()
                orig_crs = CRS.from_user_input(src.crs)
                t_to_local = Transformer.from_crs(orig_crs, src_local_crs, always_xy=True)

                b = src.bounds
                cx, cy = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
                lcx, lcy = t_to_local.transform(cx, cy)

                if orig_crs.is_geographic:
                    _, _, dx_m = geod.inv(cx, cy, cx + abs(src.transform.a), cy)
                    _, _, dy_m = geod.inv(cx, cy, cx, cy + abs(src.transform.e))
                    px_w = dx_m if src.transform.a >= 0 else -dx_m
                    px_h = -dy_m if src.transform.e <= 0 else dy_m
                else:
                    px_w = src.transform.a
                    px_h = src.transform.e

                ul_x = lcx - (src.width / 2.0) * px_w
                ul_y = lcy - (src.height / 2.0) * px_h
                new_transform = Affine(px_w, 0.0, ul_x, 0.0, px_h, ul_y)

                profile.update(crs=dst_local_crs.to_wkt(), transform=new_transform)
                with rasterio.open(dst_p, "w", **profile) as dst:
                    dst.write(data)
        else:
            shutil.copy2(p, dst_p)

    return {
        "mode": "polar_high_latitude_relocation",
        "src_center": [round(lon0, 6), round(lat0, 6)],
        "target_center": [round(target_lon, 6), round(target_lat, 6)],
    }

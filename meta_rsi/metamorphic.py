"""Label-free metamorphic invariance perturbations for silent-failure detection.

Given ONLY a task input workspace (without any ground truth or family metadata),
produces a geometrically equivalent perturbed workspace T(work_dir):
  - Local-extent datasets (span < 5 deg):
      Map WGS84 -> AEQD(center) -> AEQD(high_lat=58.3, zone_edge_lon=6.0) -> WGS84,
      preserving local metric geometry to < 0.0001% while breaking Web Mercator
      (1.9x length / 3.6x area distortion), fixed UTM zones, degree-based NN search,
      and lat/lon swaps. Also shuffles row order to expose index-alignment bugs.
  - Large-extent datasets (span >= 5 deg):
      Exact WGS84 ellipsoidal isometry: shift longitude by >= 60 deg (staying clear of
      the antimeridian) and mirror latitude (lat -> -lat, re-orienting polygon rings),
      preserving geodesic area and length to machine precision while breaking fixed
      UTM zones, hemisphere assumptions, and row-order/index bugs.
  - Optional CRS storage perturbation:
      For local GeoPackage layers, flip between WGS84 and target local UTM storage to
      catch scripts that ignore layer CRS or assume EPSG:4326 unconditionally.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.affinity
from pyproj import CRS, Geod, Transformer

SPATIAL_EXTS = {".geojson", ".gpkg", ".csv"}
RASTER_EXTS = {".tif", ".tiff"}
_WGS84_GEOD = Geod(ellps="WGS84")


def _aeqd_crs(lat: float, lon: float) -> CRS:
    """Self-contained local Azimuthal Equidistant CRS (no benchmark package import)."""
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def _geodesic_rotate_relocate(geom, cx: float, cy: float, tx: float, ty: float, rot_deg: float = 90.0):
    """Near-exact WGS84 geodesic polar transfer + bearing rotation around centroid."""
    def f(coords):
        lons, lats = coords[:, 0], coords[:, 1]
        az12, _, dist = _WGS84_GEOD.inv(
            np.full_like(lons, cx, dtype=float),
            np.full_like(lats, cy, dtype=float),
            lons,
            lats,
        )
        nlons, nlats, _ = _WGS84_GEOD.fwd(
            np.full_like(lons, tx, dtype=float),
            np.full_like(lats, ty, dtype=float),
            az12 + rot_deg,
            dist,
        )
        return np.column_stack([nlons, nlats])

    return shapely.transform(geom, f)


def _load_workspace_layers(work_dir: Path) -> list[tuple[Path, str, gpd.GeoDataFrame, CRS | None]]:
    """Return list of (path, kind, gdf_in_4326, original_crs)."""
    items = []
    for p in sorted(work_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in SPATIAL_EXTS:
            continue
        ext = p.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(p)
            if not {"lon", "lat"}.issubset(df.columns):
                continue
            gdf = gpd.GeoDataFrame(
                df.drop(columns=["lon", "lat"]),
                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                crs="EPSG:4326",
            )
            items.append((p, "csv", gdf, CRS.from_epsg(4326)))
        else:
            gdf = gpd.read_file(p)
            orig_crs = gdf.crs
            gdf4326 = gdf.to_crs("EPSG:4326") if orig_crs is not None else gdf.set_crs("EPSG:4326")
            items.append((p, ext.lstrip("."), gdf4326, orig_crs))
    return items


def _transform_geoms(geoms, tr: Transformer):
    def f(coords):
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return [shapely.transform(g, f) for g in geoms]


def _orient_if_polygonal(g):
    if g is None or g.is_empty:
        return g
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return shapely.orient_polygons(g)
    return g


def _perturb_tabular_workspace(
    src_dir: Path,
    dst_dir: Path,
    *,
    shuffle_rows: bool = True,
    seed: int = 42,
) -> dict:
    """Label-free metamorphic perturbation for general non-spatial CSV tabular workspaces:
    shuffles row order, permutes column order, and appends an unused column containing NaNs
    (catches missing sort_values, positional iloc indexing, and indiscriminate df.dropna()).
    """
    rng = np.random.default_rng(seed)
    n_files = 0
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".csv":
            shutil.copy2(p, dst_dir / p.name)
            continue
        df = pd.read_csv(p)
        if shuffle_rows and len(df) > 1:
            df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        cols = list(df.columns)
        rng.shuffle(cols)
        df = df[cols]
        # Add an unrelated audit column with NaNs (strictly label-preserving for any column-named script).
        df["_meta_probe_unused"] = [np.nan if m else "meta" for m in (rng.random(len(df)) < 0.7)]
        df.to_csv(dst_dir / p.name, index=False)
        n_files += 1
    return {"mode": "tabular_shuffle_reorder_nancol", "files": n_files}


def _relocate_geotiff(
    src_path: Path,
    dst_path: Path,
    *,
    src_lat: float,
    src_lon: float,
    target_lat: float,
    target_lon: float,
    flip_crs_and_nodata: bool = False,
) -> dict:
    """Isometrically transport a GeoTIFF from (src_lat, src_lon) to (target_lat, target_lon) via local AEQD."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    src_aeqd = _aeqd_crs(src_lat, src_lon)
    dst_aeqd = _aeqd_crs(target_lat, target_lon)

    with rasterio.open(src_path) as src:
        orig_crs = src.crs
        orig_nodata = float(src.nodata) if src.nodata is not None else -9999.0
        # 1. Warp into source local AEQD metres (near-exact metric grid)
        tr_aeqd, w_a, h_a = calculate_default_transform(
            orig_crs, src_aeqd, src.width, src.height, *src.bounds
        )
        arr_aeqd = np.full((h_a, w_a), np.float32(orig_nodata), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=arr_aeqd,
            src_transform=src.transform,
            src_crs=orig_crs,
            src_nodata=orig_nodata,
            dst_transform=tr_aeqd,
            dst_crs=src_aeqd,
            dst_nodata=orig_nodata,
            resampling=Resampling.bilinear,
        )

    # 2. Choose target storage CRS & nodata sentinel
    if abs(target_lat) >= 84.0:
        target_crs = CRS.from_epsg(3413 if target_lat >= 0 else 3031) if not flip_crs_and_nodata else CRS.from_epsg(4326)
    elif flip_crs_and_nodata:
        target_crs = CRS.from_epsg(4326) if (orig_crs is None or not orig_crs.is_geographic) else CRS.from_epsg(3857)
    else:
        target_crs = CRS.from_epsg(3857)

    new_nodata = (-32768.0 if abs(orig_nodata + 9999.0) < 1e-3 else -9999.0) if flip_crs_and_nodata else orig_nodata

    # 3. Re-tag (arr_aeqd, tr_aeqd) to dst_aeqd (zero-distortion isometric transport) and project to target_crs
    left = tr_aeqd.c
    top = tr_aeqd.f
    right = left + w_a * tr_aeqd.a
    bottom = top + h_a * tr_aeqd.e
    tr_out, w_o, h_o = calculate_default_transform(
        dst_aeqd, target_crs, w_a, h_a, left, bottom, right, top
    )
    arr_out = np.full((h_o, w_o), np.float32(new_nodata), dtype=np.float32)
    reproject(
        source=arr_aeqd,
        destination=arr_out,
        src_transform=tr_aeqd,
        src_crs=dst_aeqd,
        src_nodata=orig_nodata,
        dst_transform=tr_out,
        dst_crs=target_crs,
        dst_nodata=new_nodata,
        resampling=Resampling.bilinear,
    )

    with rasterio.open(
        dst_path,
        "w",
        driver="GTiff",
        height=h_o,
        width=w_o,
        count=1,
        dtype="float32",
        crs=target_crs,
        transform=tr_out,
        nodata=float(new_nodata),
        compress="deflate",
    ) as dst:
        dst.write(arr_out, 1)

    return {"dst_crs": target_crs.to_string(), "nodata": new_nodata}


def perturb_workspace(
    src_dir: Path,
    dst_dir: Path,
    *,
    target_lat: float = 58.3,
    target_lon: float = 6.0,
    shuffle_rows: bool = True,
    flip_gpkg_crs: bool = False,
    seed: int = 42,
) -> dict:
    """Create a label-preserving perturbed copy of `src_dir` inside `dst_dir`.

    Automatically dispatches to spatial AEQD/isometry perturbations for vector and/or GeoTIFF
    geospatial workspaces, or tabular shuffle/column-reorder/unused-NaN-column perturbations
    for general CSV workspaces.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    layers = _load_workspace_layers(src_dir)
    rasters = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in RASTER_EXTS)
    if not layers and not rasters:
        return _perturb_tabular_workspace(src_dir, dst_dir, shuffle_rows=shuffle_rows, seed=seed)

    # Copy any non-spatial auxiliary files verbatim.
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() not in (SPATIAL_EXTS | RASTER_EXTS):
            shutil.copy2(p, dst_dir / p.name)

    # Compute spherical circular centroid (exact across antimeridian and for raster-only workspaces)
    xs_u, ys_u, zs_u = [], [], []
    wgs_bounds = []
    for _, _, g, _ in layers:
        if not g.empty:
            wgs_bounds.append(g.total_bounds)
            pts = g.geometry.representative_point()
            rlon = np.deg2rad(pts.x.to_numpy(dtype=float))
            rlat = np.deg2rad(pts.y.to_numpy(dtype=float))
            clat = np.cos(rlat)
            xs_u.append(float(np.mean(clat * np.cos(rlon))))
            ys_u.append(float(np.mean(clat * np.sin(rlon))))
            zs_u.append(float(np.mean(np.sin(rlat))))
    if rasters:
        import rasterio
        for rp in rasters:
            with rasterio.open(rp) as src:
                b = src.bounds
                tr4326 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                l_lo, b_lo = tr4326.transform(b.left, b.bottom)
                r_hi, t_hi = tr4326.transform(b.right, b.top)
                wgs_bounds.append(np.array([min(l_lo, r_hi), min(b_lo, t_hi), max(l_lo, r_hi), max(b_lo, t_hi)]))
                clon, clat_d = tr4326.transform(0.5 * (b.left + b.right), 0.5 * (b.bottom + b.top))
                rlon, rlat = np.deg2rad(clon), np.deg2rad(clat_d)
                xs_u.append(float(np.cos(rlat) * np.cos(rlon)))
                ys_u.append(float(np.cos(rlat) * np.sin(rlon)))
                zs_u.append(float(np.sin(rlat)))

    all_bounds = np.array(wgs_bounds)
    minx, miny = float(all_bounds[:, 0].min()), float(all_bounds[:, 1].min())
    maxx, maxy = float(all_bounds[:, 2].max()), float(all_bounds[:, 3].max())
    mx, my, mz = float(np.mean(xs_u)), float(np.mean(ys_u)), float(np.mean(zs_u))
    cy = float(np.degrees(np.arctan2(mz, np.hypot(mx, my))))
    cx = float(np.degrees(np.arctan2(my, mx)))
    span_lon = maxx - minx
    span_lat = maxy - miny
    # Antimeridian wraparound (span_lon > 180) is still a local dataset!
    if span_lon > 180.0:
        span_lon = 360.0 - span_lon
    is_large = (not rasters) and (max(span_lon, span_lat) >= 5.0)

    rng = np.random.default_rng(seed)

    if not is_large:
        tr_to_plane = Transformer.from_crs("EPSG:4326", _aeqd_crs(cy, cx), always_xy=True)
        tr_to_target = Transformer.from_crs(_aeqd_crs(target_lat, target_lon), "EPSG:4326", always_xy=True)
        mode = "local_aeqd_shift"
        meta = {
            "mode": mode,
            "src_center": [round(cy, 3), round(cx, 3)],
            "dst_center": [target_lat, target_lon],
            "span_deg": [round(span_lat, 3), round(span_lon, 3)],
        }
    else:
        # Non-6-degree-multiple longitude shift + 90-degree geodesic bearing rotation around centroid
        # (breaks both Web Mercator and single-zone UTM distortion while preserving WGS84 geodesic
        # area/length to <0.01%). Falls back to lat-mirror + non-6-multiple shift if bounds exceed safe box.
        cand_tx = []
        for s in (73.7, -73.7, 51.3, -51.3, 31.7, -31.7, 15.3, -15.3):
            tx = cx + s
            if -140.0 <= tx <= 140.0:
                cand_tx.append(tx)
        cand_tx.append(cx)
        rot_plan = None
        for tx in cand_tx:
            for ty in (28.0, -28.0, 0.0):
                for rot_deg in (90.0, -90.0):
                    ok_box = True
                    trial_geoms = []
                    for _, _, gdf4326, _ in layers:
                        ng = [_orient_if_polygonal(_geodesic_rotate_relocate(g, cx, cy, tx, ty, rot_deg)) for g in gdf4326.geometry]
                        tmp_g = gpd.GeoSeries(ng, crs="EPSG:4326")
                        tb = tmp_g.total_bounds
                        if tb[0] < -172.0 or tb[2] > 172.0 or tb[1] < -78.0 or tb[3] > 78.0:
                            ok_box = False
                            break
                        trial_geoms.append(ng)
                    if ok_box:
                        rot_plan = (tx, ty, rot_deg, trial_geoms)
                        break
                if rot_plan is not None:
                    break
            if rot_plan is not None:
                break

        if rot_plan is not None:
            tx, ty, rot_deg, large_rotated_geoms = rot_plan
            mode = "large_geodesic_rotation"
            meta = {
                "mode": mode,
                "src_center": [round(cy, 3), round(cx, 3)],
                "dst_center": [round(ty, 3), round(tx, 3)],
                "rot_deg": rot_deg,
                "span_deg": [round(span_lat, 3), round(span_lon, 3)],
            }
        else:
            large_rotated_geoms = None
            cand_shifts = [73.7, -73.7, 51.3, -51.3, 31.7, -31.7]
            lon_shift = 0.0
            for s in cand_shifts:
                if minx + s > -172.0 and maxx + s < 172.0:
                    lon_shift = s
                    break
            mode = "large_ellipsoidal_isometry"
            meta = {
                "mode": mode,
                "src_center": [round(cy, 3), round(cx, 3)],
                "lon_shift": lon_shift,
                "lat_mirror": True,
                "span_deg": [round(span_lat, 3), round(span_lon, 3)],
            }

    # For local projected GPKG output, compute one shared target UTM CRS if needed.
    transformed: list[tuple[Path, str, gpd.GeoDataFrame, CRS | None]] = []
    for idx, (path, kind, gdf4326, orig_crs) in enumerate(layers):
        if not is_large:
            plane_geoms = _transform_geoms(list(gdf4326.geometry), tr_to_plane)
            new_geoms = _transform_geoms(plane_geoms, tr_to_target)
        elif large_rotated_geoms is not None:
            new_geoms = large_rotated_geoms[idx]
        else:
            new_geoms = [
                _orient_if_polygonal(
                    shapely.affinity.translate(
                        shapely.affinity.scale(g, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)),
                        xoff=lon_shift,
                        yoff=0.0,
                    )
                )
                for g in gdf4326.geometry
            ]
        g_new = gpd.GeoDataFrame(gdf4326.drop(columns="geometry").copy(), geometry=new_geoms, crs="EPSG:4326")
        if shuffle_rows and len(g_new) > 1:
            perm = rng.permutation(len(g_new))
            g_new = g_new.iloc[perm].reset_index(drop=True)
        transformed.append((path, kind, g_new, orig_crs))

    for path, kind, g_new, orig_crs in transformed:
        out_path = dst_dir / path.name
        if kind == "csv":
            df = pd.DataFrame(g_new.drop(columns="geometry"))
            df["lon"] = g_new.geometry.x.round(7)
            df["lat"] = g_new.geometry.y.round(7)
            df.to_csv(out_path, index=False)
        elif kind == "gpkg":
            orig_is_projected = orig_crs is not None and not orig_crs.is_geographic
            store_utm = (not orig_is_projected) if (flip_gpkg_crs and not is_large) else (orig_is_projected and not is_large)
            if store_utm:
                try:
                    dst_crs = g_new.estimate_utm_crs()
                except RuntimeError:
                    # Outside UTM latitude bounds (>84N / <80S): fall back to standard Polar Stereographic.
                    dst_crs = CRS.from_epsg(3413 if target_lat >= 0 else 3031)
                g_out = g_new.to_crs(dst_crs)
            else:
                g_out = g_new
            g_out.to_file(out_path, driver="GPKG")
        else:
            g_new.to_file(out_path, driver="GeoJSON")

    if rasters:
        raster_meta = {}
        for rp in rasters:
            raster_meta[rp.name] = _relocate_geotiff(
                rp,
                dst_dir / rp.name,
                src_lat=cy,
                src_lon=cx,
                target_lat=target_lat,
                target_lon=target_lon,
                flip_crs_and_nodata=flip_gpkg_crs,
            )
        meta["rasters"] = raster_meta

    meta["files"] = len(transformed) + len(rasters)
    return meta

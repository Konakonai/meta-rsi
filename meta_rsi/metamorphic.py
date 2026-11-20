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
from pyproj import CRS, Transformer

from benchmarks.gsb.geo import local_crs

SPATIAL_EXTS = {".geojson", ".gpkg", ".csv"}


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

    Automatically dispatches to spatial AEQD/isometry perturbations for geospatial workspaces,
    or tabular shuffle/column-reorder/unused-NaN-column perturbations for general CSV workspaces.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    layers = _load_workspace_layers(src_dir)
    if not layers:
        return _perturb_tabular_workspace(src_dir, dst_dir, shuffle_rows=shuffle_rows, seed=seed)

    # Copy any non-spatial auxiliary files verbatim.
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() not in SPATIAL_EXTS:
            shutil.copy2(p, dst_dir / p.name)

    all_bounds = np.array([g.total_bounds for _, _, g, _ in layers])
    minx, miny = float(all_bounds[:, 0].min()), float(all_bounds[:, 1].min())
    maxx, maxy = float(all_bounds[:, 2].max()), float(all_bounds[:, 3].max())
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    span_lon, span_lat = maxx - minx, maxy - miny
    is_large = max(span_lon, span_lat) >= 5.0

    rng = np.random.default_rng(seed)

    if not is_large:
        tr_to_plane = Transformer.from_crs("EPSG:4326", local_crs(cy, cx), always_xy=True)
        tr_to_target = Transformer.from_crs(local_crs(target_lat, target_lon), "EPSG:4326", always_xy=True)
        mode = "local_aeqd_shift"
        meta = {
            "mode": mode,
            "src_center": [round(cy, 3), round(cx, 3)],
            "dst_center": [target_lat, target_lon],
            "span_deg": [round(span_lat, 3), round(span_lon, 3)],
        }
    else:
        # Choose a clean longitude shift keeping all coordinates safely inside [-172, 172].
        cand_shifts = [72.0, -72.0, 54.0, -54.0, 36.0, -36.0]
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
    for path, kind, gdf4326, orig_crs in layers:
        if not is_large:
            plane_geoms = _transform_geoms(list(gdf4326.geometry), tr_to_plane)
            new_geoms = _transform_geoms(plane_geoms, tr_to_target)
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

    meta["files"] = len(transformed)
    return meta

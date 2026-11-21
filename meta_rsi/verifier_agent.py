"""Non-Circular Algebraic/Geometric Self-Certified Verifier Co-Evolution (`verifier_agent.py`).

Addresses the fundamental limitation of fixed human-written metamorphic operators AND
avoids Double Ratchet's reliance on labeled human anchor sets (arXiv:2607.12790) or
Verifier Hardening's reliance on per-task gold reference solutions (arXiv:2610.15924).

Pipeline:
  1. Verifier-Proposer inspects scripts emitted by a converged solver harness that already
     pass the current invariance set T_k (e.g., mid-latitude perturbations).
  2. Proposer synthesizes a candidate Python perturbation operator:
         def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict
  3. Zero-Label Self-Certification Gate verifies 3 strictly non-circular properties:
     (a) AST Safety: no network, eval/exec, subprocess, or shell escape.
     (b) Algebraic/Geometric Conservation on unlabeled dev workspaces (x, T(x)):
         - Exact file list, layer count, column sets, row counts, and attribute multisets.
         - Valid WGS84 bounds & non-empty finite geometries.
         - Pairwise WGS84 ellipsoidal geodesic distances (`pyproj.Geod`) match < 0.1%
           (row-order invariant via sorted pairwise distance spectrum).
         - Total WGS84 geodesic polygon areas & linestring lengths match < 0.1%.
         - Cross-layer topological relation counts (`intersects`, `within`) match 100%.
     (c) Non-Vacuity & Active Discrimination:
         - Transform is non-trivial (centroid moves >100 km OR bearing rotates >15 deg OR CRS flips).
         - Exposes a crash or >1% numeric drift on >=1 currently passing solver script.
  4. Only certified operators enter T_{k+1} to drive the next stage of solver evolution!
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Geod

from .llm import LLMClient
from .meta_agent import _extract_fenced_blocks, _stratified_pick, _strip_thinking_preamble
from .sandbox import FORBIDDEN_CALLS, FORBIDDEN_IMPORTS, parse_final_number, run_python_sandbox

_GEOD = Geod(ellps="WGS84")
_SPATIAL_EXTS = {".geojson", ".gpkg", ".csv"}
_RASTER_EXTS = {".tif", ".tiff"}

VERIFIER_SYSTEM_PROMPT = """You are an AI verification researcher co-evolving label-free metamorphic invariance probes
for scientific data-analysis coding agents.

You will be shown Python scripts emitted by a solver harness that ALREADY pass the existing invariance
probe set T_k, along with the workspace file types. Your job is to inspect the passing scripts closely,
identify ONE untested input transformation or representation change where those scripts rely on an
implicit assumption that is NOT guaranteed to hold on equivalent inputs, and write ONE self-contained
Python perturbation module implementing a mathematically invariant transformation `T(x)` that exposes
this fragility WITHOUT changing the true task answer.

Hard contract for your Python module:
1. Must define: `def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict`
2. Allowed imports: stdlib (`pathlib`, `shutil`, `math`, `random`, `json`, `csv`) plus `geopandas`, `pyproj`, `shapely`,
   `numpy`, `pandas`, `rasterio`.
3. Must copy or transform ALL files from `src_dir` to `dst_dir` (creating `dst_dir` if needed), preserving
   exact filenames, layer/table counts, column names, row counts, and attribute value multisets.
4. Keep all output coordinates finite and valid in their declared CRS. Keep `.csv` coordinate columns
   rounded to 7 decimals. If transforming raster files alongside vector layers, transform all files in the
   workspace consistently so spatial alignment and valid-data distributions are preserved.
5. Must return a metadata dict `{"mode": "<short_name>", ...}`.
6. Your operator will be automatically audited without ground-truth labels for:
   - AST safety and clean execution
   - Exact schema, row count, and attribute multiset preservation
   - Exact geometric conservation (pairwise ellipsoidal distances, polygon areas, line lengths, cross-layer
     topological relations) and raster physical-area / value-distribution conservation when rasters are present
   - Non-vacuity (non-trivial input/representation change) AND active failure/drift exposure on at least
     one currently passing script.

Output format:
HYPOTHESIS: <2 sentences naming the implicit assumption in the passing scripts and how your invariant transformation exposes it>

```python
<complete self-contained python module defining perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict>
```
"""


def validate_probe_source(source: str) -> tuple[bool, str]:
    """AST security and contract check for candidate perturbation modules."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    has_perturb = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                return False, f"Forbidden import-from: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, f"Forbidden call: {node.func.id}()"
        elif isinstance(node, ast.FunctionDef) and node.name == "perturb":
            if len(node.args.args) >= 2:
                has_perturb = True
    if not has_perturb:
        return False, "Missing `def perturb(src_dir, dst_dir, seed=42)`"
    return True, "ok"


def load_probe_fn(source: str, tmp_dir: Path, name: str = "cand_probe") -> Callable:
    ok, reason = validate_probe_source(source)
    if not ok:
        raise ValueError(f"Probe AST validation failed: {reason}")
    p = tmp_dir / f"{name}.py"
    p.write_text(source)
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probe spec from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.perturb


def _load_layers_for_audit(wdir: Path) -> dict[str, tuple[gpd.GeoDataFrame, CRS | None]]:
    out: dict[str, tuple[gpd.GeoDataFrame, CRS | None]] = {}
    for p in sorted(wdir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _SPATIAL_EXTS:
            continue
        ext = p.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(p)
            if not {"lon", "lat"}.issubset(df.columns):
                continue
            g = gpd.GeoDataFrame(
                df.drop(columns=["lon", "lat"]),
                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                crs="EPSG:4326",
            )
            out[p.name] = (g, CRS.from_epsg(4326))
        else:
            g = gpd.read_file(p)
            orig_crs = g.crs
            g4326 = g.to_crs("EPSG:4326") if orig_crs is not None else g.set_crs("EPSG:4326")
            out[p.name] = (g4326, orig_crs)
    return out


def _geod_area_length(g4326: gpd.GeoDataFrame) -> tuple[float, float]:
    tot_area = 0.0
    tot_len = 0.0
    for geom in g4326.geometry:
        if geom is None or geom.is_empty:
            continue
        gt = geom.geom_type
        if gt in ("Polygon", "MultiPolygon"):
            a, _ = _GEOD.geometry_area_perimeter(geom)
            tot_area += abs(a)
        elif gt in ("LineString", "MultiLineString"):
            tot_len += abs(_GEOD.geometry_length(geom))
    return tot_area, tot_len


def _pairwise_geod_spectrum(g4326: gpd.GeoDataFrame, max_pts: int = 25) -> np.ndarray:
    """Row-order-invariant sorted pairwise WGS84 geodesic distances among feature centroids."""
    # Sort deterministically by non-geometry attribute values so sampling is identical even after row shuffle
    attr_cols = [c for c in g4326.columns if c != "geometry"]
    if attr_cols:
        key_series = g4326[attr_cols].astype(str).agg("|".join, axis=1)
        order = np.argsort(key_series.to_numpy(dtype=str), kind="mergesort")
        sub = g4326.iloc[order]
    else:
        sub = g4326
    if len(sub) > max_pts:
        idx = np.linspace(0, len(sub) - 1, max_pts, dtype=int)
        sub = sub.iloc[idx]
    if len(sub) < 2:
        return np.array([], dtype=float)
    # Use a true representative vertex per geometry (invariant under smooth coordinate transforms,
    # unlike WGS84 degree bbox midpoints which warp near the poles due to meridian convergence).
    xs, ys = [], []
    for geom in sub.geometry:
        g0 = geom.geoms[0] if hasattr(geom, "geoms") else geom
        if g0.geom_type == "Point":
            xs.append(g0.x)
            ys.append(g0.y)
        elif g0.geom_type == "Polygon":
            x0, y0 = g0.exterior.coords[0][:2]
            xs.append(x0)
            ys.append(y0)
        else:
            x0, y0 = g0.coords[0][:2]
            xs.append(x0)
            ys.append(y0)
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    i_idx, j_idx = np.triu_indices(len(xs_arr), k=1)
    _, _, dists = _GEOD.inv(xs_arr[i_idx], ys_arr[i_idx], xs_arr[j_idx], ys_arr[j_idx])
    return np.sort(np.asarray(dists, dtype=float))


def _circular_centroid_gdf(g4326: gpd.GeoDataFrame) -> tuple[float, float]:
    """Spherical unit-vector centroid (lon, lat) in degrees, exact across +/-180 deg."""
    if g4326.empty:
        return 0.0, 0.0
    pts = g4326.geometry.representative_point()
    rlon = np.deg2rad(pts.x.to_numpy(dtype=float))
    rlat = np.deg2rad(pts.y.to_numpy(dtype=float))
    clat = np.cos(rlat)
    mx = float(np.mean(clat * np.cos(rlon)))
    my = float(np.mean(clat * np.sin(rlon)))
    mz = float(np.mean(np.sin(rlat)))
    return float(np.degrees(np.arctan2(my, mx))), float(np.degrees(np.arctan2(mz, np.hypot(mx, my))))


def _circular_centroid_layers(layers: dict[str, tuple[gpd.GeoDataFrame, CRS | None]]) -> tuple[float, float]:
    xs, ys, zs = [], [], []
    for g, _ in layers.values():
        if g.empty:
            continue
        lon, lat = _circular_centroid_gdf(g)
        rlon, rlat = np.deg2rad(lon), np.deg2rad(lat)
        xs.append(float(np.cos(rlat) * np.cos(rlon)))
        ys.append(float(np.cos(rlat) * np.sin(rlon)))
        zs.append(float(np.sin(rlat)))
    if not xs:
        return 0.0, 0.0
    mx, my, mz = float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))
    return float(np.degrees(np.arctan2(my, mx))), float(np.degrees(np.arctan2(mz, np.hypot(mx, my))))


def _raster_audit_stats(path: Path) -> dict:
    """Compute zero-label physical-area & valid-value distribution invariants for a GeoTIFF."""
    import rasterio
    from pyproj import Transformer
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(path) as src:
        orig_crs = src.crs
        nodata = float(src.nodata) if src.nodata is not None else -9999.0
        b = src.bounds
        tr4326 = Transformer.from_crs(orig_crs, "EPSG:4326", always_xy=True)
        clon, clat = tr4326.transform(0.5 * (b.left + b.right), 0.5 * (b.bottom + b.top))
        aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m +no_defs")
        tr_m, w_m, h_m = calculate_default_transform(orig_crs, aeqd, src.width, src.height, *b, resolution=50.0)
        if w_m <= 0 or h_m <= 0 or (w_m * h_m) > 25_000_000:
            raise ValueError(f"Invalid reprojected raster grid size: {w_m}x{h_m}")
        arr = np.full((h_m, w_m), np.float32(nodata), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=orig_crs,
            src_nodata=nodata,
            dst_transform=tr_m,
            dst_crs=aeqd,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
    valid = np.isfinite(arr) & (~np.isclose(arr, nodata))
    vals = arr[valid].astype(np.float64)
    pix_area = abs(tr_m.a * tr_m.e)
    return {
        "crs": orig_crs.to_string() if orig_crs else None,
        "nodata": nodata,
        "clon": float(clon),
        "clat": float(clat),
        "valid_area_m2": float(np.sum(valid) * pix_area),
        "mean": float(np.mean(vals)) if len(vals) else float("nan"),
        "std": float(np.std(vals)) if len(vals) else float("nan"),
        "q50": float(np.median(vals)) if len(vals) else float("nan"),
        "n_valid": int(len(vals)),
    }


_SI_SCALE_OFFSET: dict[str, tuple[float, float]] = {
    "m": (1.0, 0.0), "km": (1000.0, 0.0), "cm": (0.01, 0.0), "mm": (0.001, 0.0), "ft": (0.3048, 0.0), "in": (0.0254, 0.0),
    "m2": (1.0, 0.0), "cm2": (1e-4, 0.0), "ft2": (0.09290304, 0.0),
    "m3": (1.0, 0.0), "L": (1e-3, 0.0), "ft3": (0.028316846592, 0.0), "gal_us": (0.003785411784, 0.0),
    "kg": (1.0, 0.0), "g": (1e-3, 0.0), "t": (1000.0, 0.0), "lb": (0.45359237, 0.0),
    "s": (1.0, 0.0), "min": (60.0, 0.0), "hr": (3600.0, 0.0),
    "m/s": (1.0, 0.0), "km/h": (1.0 / 3.6, 0.0), "mph": (0.44704, 0.0), "ft/s": (0.3048, 0.0),
    "K": (1.0, 0.0), "degC": (1.0, 273.15), "degF": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    "Pa": (1.0, 0.0), "kPa": (1000.0, 0.0), "bar": (1e5, 0.0), "psi": (6894.757293168, 0.0),
    "N": (1.0, 0.0), "kN": (1000.0, 0.0), "lbf": (4.4482216152605, 0.0),
    "J": (1.0, 0.0), "kJ": (1000.0, 0.0), "MJ": (1e6, 0.0), "kWh": (3.6e6, 0.0), "kcal": (4184.0, 0.0),
    "W": (1.0, 0.0), "kW": (1000.0, 0.0), "MW": (1e6, 0.0), "hp": (745.6998715822702, 0.0),
    "W/m2": (1.0, 0.0), "kW/m2": (1000.0, 0.0),
    "kg/m3": (1.0, 0.0), "g/cm3": (1000.0, 0.0), "lb/ft3": (16.01846337396014, 0.0),
    "J/(kg*K)": (1.0, 0.0), "kJ/(kg*K)": (1000.0, 0.0),
    "m4": (1.0, 0.0), "cm4": (1e-8, 0.0), "in4": (0.0254 ** 4, 0.0),
    "rad": (1.0, 0.0), "deg": (float(np.pi / 180.0), 0.0),
    "g/L": (1.0, 0.0), "mg/L": (1e-3, 0.0), "kg/L": (1000.0, 0.0),
    "fraction": (1.0, 0.0), "percent": (0.01, 0.0),
}


def _audit_tabular_metadata_dir(src_dir: Path, dst_dir: Path, tol_rel: float) -> tuple[bool, str, bool]:
    """Zero-label algebraic physical-value conservation audit for tabular+metadata workspaces."""
    try:
        m0 = json.loads((src_dir / "metadata.json").read_text())
        m1 = json.loads((dst_dir / "metadata.json").read_text())
    except Exception as exc:
        return False, f"Invalid metadata.json: {exc}", False
    t0 = m0.get("tables") or {}
    t1 = m1.get("tables") or {}
    if set(t0.keys()) != set(t1.keys()):
        return False, f"metadata.json tables mismatch: {sorted(t0)} vs {sorted(t1)}", False

    rep_changed = False
    for fname in sorted(t0):
        df0 = pd.read_csv(src_dir / fname)
        df1 = pd.read_csv(dst_dir / fname)
        if len(df0) != len(df1):
            return False, f"Row count mismatch in {fname}: {len(df0)} vs {len(df1)}", False
        req_cols = [c for c in df0.columns if not c.startswith("_meta_")]
        missing_cols = [c for c in req_cols if c not in df1.columns]
        if missing_cols:
            return False, f"Missing columns in {fname}: {missing_cols}", False

        tm0, tm1 = t0[fname], t1[fname]
        if (
            tm0.get("units") != tm1.get("units")
            or tm0.get("calibration") != tm1.get("calibration")
            or tm0.get("missing_sentinel") != tm1.get("missing_sentinel")
            or list(df0.columns) != list(df1.columns)
        ):
            rep_changed = True

        def _norm_df(df: pd.DataFrame, tm: dict) -> pd.DataFrame:
            out = df.copy()
            sent = tm.get("missing_sentinel")
            if sent is not None:
                for c in req_cols:
                    if pd.api.types.is_numeric_dtype(out[c]):
                        mask = np.isclose(out[c].to_numpy(dtype=float), float(sent), atol=1e-6, equal_nan=False)
                        if np.any(mask):
                            out.loc[mask, c] = np.nan
            for c, cfg in (tm.get("calibration") or {}).items():
                if c in out.columns:
                    out[c] = out[c].astype(float) * float(cfg.get("scale", 1.0)) + float(cfg.get("offset", 0.0))
            for c, u in (tm.get("units") or {}).items():
                if c in out.columns:
                    if u not in _SI_SCALE_OFFSET:
                        raise ValueError(f"Unknown unit '{u}' in {fname}:{c}")
                    sc, off = _SI_SCALE_OFFSET[u]
                    out[c] = out[c].astype(float) * sc + off
            return out

        try:
            n0 = _norm_df(df0, tm0)
            n1 = _norm_df(df1, tm1)
        except Exception as exc:
            return False, f"Normalization error in {fname}: {exc}", False

        for c in req_cols:
            if pd.api.types.is_numeric_dtype(n0[c]):
                v0 = np.sort(n0[c].dropna().to_numpy(dtype=float))
                v1 = np.sort(n1[c].dropna().to_numpy(dtype=float))
                if len(v0) != len(v1):
                    return False, f"Non-missing count mismatch in {fname}:{c} ({len(v0)} vs {len(v1)})", False
                if len(v0) > 0:
                    # Allow constant time-origin shift on monotonic elapsed_time columns (dt-invariant)
                    if "time" in c.lower():
                        d0, d1 = np.diff(v0), np.diff(v1)
                        denom = np.maximum(np.abs(d0), 1e-6)
                        err = float(np.max(np.abs(d1 - d0) / denom)) if len(d0) else 0.0
                    else:
                        denom = np.maximum(np.abs(v0), 1e-6)
                        err = float(np.max(np.abs(v1 - v0) / denom))
                    if err > max(tol_rel, 1e-3):
                        return False, f"Physical value drift {err:.4%} > {tol_rel:.2%} in {fname}:{c}", False
            else:
                if sorted(n0[c].astype(str).tolist()) != sorted(n1[c].astype(str).tolist()):
                    return False, f"Categorical multiset mutated in {fname}:{c}", False

    return True, "ok", rep_changed


def _topo_counts(layers: dict[str, tuple[gpd.GeoDataFrame, CRS | None]]) -> dict[str, int]:
    """Exact cross-layer topological relation counts in local circular-mean AEQD."""
    names = sorted(layers.keys())
    if len(names) < 2:
        return {}
    cx, cy = _circular_centroid_layers(layers)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m +no_defs")
    proj = {k: layers[k][0].to_crs(aeqd) for k in names}
    counts: dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            ga, gb = proj[a_name], proj[b_name]
            for pred in ("intersects", "within"):
                try:
                    c = len(gpd.sjoin(ga[["geometry"]], gb[["geometry"]], predicate=pred))
                except Exception:
                    c = -1
                counts[f"{a_name}__{b_name}__{pred}"] = c
    return counts


def certify_perturbation_operator(
    probe_source: str,
    dev_dirs: list[Path],
    passing_cases: list[tuple[Path, str, float, str]],
    *,
    tol_rel: float = 1e-3,
    min_displacement_m: float = 100_000.0,
    timeout: float = 25.0,
    python: str = sys.executable,
) -> dict:
    """Zero-label, non-circular self-certification audit for a proposed operator T(x).

    `passing_cases`: list of `(work_dir, code, orig_val, answer_type)` for solver scripts
    that currently pass T_k (used ONLY for the anti-vacuity active-discrimination check).
    """
    ok_ast, ast_reason = validate_probe_source(probe_source)
    if not ok_ast:
        return {"certified": False, "stage": "ast", "reason": ast_reason}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            probe_fn = load_probe_fn(probe_source, tmp_path)
        except Exception as exc:
            return {"certified": False, "stage": "load", "reason": f"{type(exc).__name__}: {exc}"}

        max_dist_err = 0.0
        max_area_err = 0.0
        max_len_err = 0.0
        max_disp_m = 0.0
        crs_changed = False
        topo_mismatches = 0
        audited_dirs = 0

        for idx, src_dir in enumerate(dev_dirs):
            dst_dir = tmp_path / f"audit_{idx}"
            orig_files = sorted(p.name for p in src_dir.iterdir() if p.is_file())
            try:
                probe_fn(src_dir, dst_dir, 42 + idx)
            except Exception as exc:
                return {"certified": False, "stage": "exec", "reason": f"perturb() crashed on workspace #{idx} (files={orig_files}): {type(exc).__name__}: {exc}"}

            pert_files = sorted(p.name for p in dst_dir.iterdir() if p.is_file()) if dst_dir.exists() else []
            if orig_files != pert_files:
                return {"certified": False, "stage": "schema", "reason": f"File list mismatch: {orig_files} vs {pert_files}"}

            orig_layers = _load_layers_for_audit(src_dir)
            pert_layers = _load_layers_for_audit(dst_dir)
            if set(orig_layers.keys()) != set(pert_layers.keys()):
                return {"certified": False, "stage": "schema", "reason": "Spatial layer keys mismatch"}

            ws_span_deg = 0.0
            for g_o, _ in orig_layers.values():
                if not g_o.empty:
                    tb_o = g_o.total_bounds
                    slon = float(tb_o[2] - tb_o[0])
                    if slon > 180.0:
                        slon = 360.0 - slon
                    ws_span_deg = max(ws_span_deg, slon, float(tb_o[3] - tb_o[1]))
            ws_tag = f"workspace #{idx} (span={ws_span_deg:.1f}deg, files={orig_files})"

            for fname, (g_orig, crs_orig) in orig_layers.items():
                g_pert, crs_pert = pert_layers[fname]
                if len(g_orig) != len(g_pert):
                    return {"certified": False, "stage": "schema", "reason": f"Row count mismatch in {fname} on {ws_tag}"}
                cols_o = sorted(c for c in g_orig.columns if c != "geometry")
                cols_p = sorted(c for c in g_pert.columns if c != "geometry")
                if cols_o != cols_p:
                    return {"certified": False, "stage": "schema", "reason": f"Column mismatch in {fname}: {cols_o} vs {cols_p}"}
                # Attribute multiset preservation (row-order invariant)
                if cols_o:
                    ms_o = sorted(g_orig[cols_o].astype(str).agg("|".join, axis=1).tolist())
                    ms_p = sorted(g_pert[cols_o].astype(str).agg("|".join, axis=1).tolist())
                    if ms_o != ms_p:
                        return {"certified": False, "stage": "schema", "reason": f"Attribute multiset mutated in {fname} on {ws_tag}"}
                # Bounds sanity
                if g_pert.geometry.isna().any() or g_pert.geometry.is_empty.any():
                    return {"certified": False, "stage": "geometry", "reason": f"Empty/NaN geometry in {fname} on {ws_tag}"}
                tb = g_pert.total_bounds
                if not np.all(np.isfinite(tb)) or tb[0] < -180.0 or tb[2] > 180.0 or tb[1] < -90.0 or tb[3] > 90.0:
                    return {"certified": False, "stage": "geometry", "reason": f"Invalid WGS84 bounds {tb} in {fname} on {ws_tag}"}

                if crs_orig != crs_pert:
                    crs_changed = True

                # Centroid displacement (antimeridian-safe circular centroid)
                ocx, ocy = _circular_centroid_gdf(g_orig)
                pcx, pcy = _circular_centroid_gdf(g_pert)
                _, _, disp = _GEOD.inv(ocx, ocy, pcx, pcy)
                max_disp_m = max(max_disp_m, float(disp))

                # Geodesic area & length conservation
                a0, l0 = _geod_area_length(g_orig)
                a1, l1 = _geod_area_length(g_pert)
                if a0 > 1.0:
                    ae = abs(a1 - a0) / a0
                    max_area_err = max(max_area_err, ae)
                    if ae > tol_rel:
                        return {"certified": False, "stage": "conservation", "reason": f"Geodesic area error {ae:.4%} > {tol_rel:.2%} in {fname} on {ws_tag}"}
                if l0 > 1.0:
                    le = abs(l1 - l0) / l0
                    max_len_err = max(max_len_err, le)
                    if le > tol_rel:
                        return {"certified": False, "stage": "conservation", "reason": f"Geodesic length error {le:.4%} > {tol_rel:.2%} in {fname} on {ws_tag}"}

                # Pairwise geodesic distance spectrum conservation
                d0 = _pairwise_geod_spectrum(g_orig)
                d1 = _pairwise_geod_spectrum(g_pert)
                if len(d0) > 0:
                    mask = d0 > 10.0
                    if np.any(mask):
                        de = float(np.max(np.abs(d1[mask] - d0[mask]) / d0[mask]))
                        max_dist_err = max(max_dist_err, de)
                        if de > max(tol_rel * 3.0, 3e-3):
                            return {"certified": False, "stage": "conservation", "reason": f"Pairwise geodesic dist error {de:.4%} in {fname} on {ws_tag}"}

            # Cross-layer topological conservation
            t0 = _topo_counts(orig_layers)
            t1 = _topo_counts(pert_layers)
            if t0 != t1:
                topo_mismatches += 1
                return {"certified": False, "stage": "topology", "reason": f"Topological counts changed on {ws_tag}: {t0} vs {t1}"}

            # GeoTIFF raster conservation (when rasters are present)
            orig_rasters = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in _RASTER_EXTS)
            orig_vec_center = _circular_centroid_layers(orig_layers) if orig_layers else None
            pert_vec_center = _circular_centroid_layers(pert_layers) if pert_layers else None
            for rp in orig_rasters:
                pp = dst_dir / rp.name
                try:
                    rs0 = _raster_audit_stats(rp)
                    rs1 = _raster_audit_stats(pp)
                except Exception as exc:
                    return {"certified": False, "stage": "raster", "reason": f"Failed reading raster {rp.name}: {exc}"}
                if rs1["n_valid"] == 0 or not np.isfinite(rs1["mean"]):
                    return {"certified": False, "stage": "raster", "reason": f"Perturbed raster {rp.name} has no valid pixels"}
                if orig_vec_center is not None and pert_vec_center is not None:
                    _, _, vr_d0 = _GEOD.inv(orig_vec_center[0], orig_vec_center[1], rs0["clon"], rs0["clat"])
                    _, _, vr_d1 = _GEOD.inv(pert_vec_center[0], pert_vec_center[1], rs1["clon"], rs1["clat"])
                    if abs(vr_d1 - vr_d0) > max(500.0, 0.10 * vr_d0):
                        return {
                            "certified": False,
                            "stage": "conservation",
                            "reason": f"Vector-raster spatial alignment broken for {rp.name}: center separation changed from {vr_d0:.0f}m to {vr_d1:.0f}m (transform rasters alongside vectors or copy raster workspaces verbatim).",
                        }
                if rs0["crs"] != rs1["crs"] or abs(rs0["nodata"] - rs1["nodata"]) > 1e-6:
                    crs_changed = True
                _, _, rdisp = _GEOD.inv(rs0["clon"], rs0["clat"], rs1["clon"], rs1["clat"])
                max_disp_m = max(max_disp_m, float(rdisp))
                # Physical footprint area and valid-pixel distribution conservation (<1.5% allowing resampling edge)
                r_tol = max(tol_rel * 15.0, 0.015)
                if rs0["valid_area_m2"] > 1.0:
                    rae = abs(rs1["valid_area_m2"] - rs0["valid_area_m2"]) / rs0["valid_area_m2"]
                    max_area_err = max(max_area_err, rae)
                    if rae > r_tol:
                        return {"certified": False, "stage": "conservation", "reason": f"Raster valid area error {rae:.4%} > {r_tol:.2%} in {rp.name}"}
                for stat_k in ("mean", "q50"):
                    denom = max(abs(rs0[stat_k]), 1e-6)
                    rse = abs(rs1[stat_k] - rs0[stat_k]) / denom
                    if rse > r_tol:
                        return {"certified": False, "stage": "conservation", "reason": f"Raster {stat_k} drift {rse:.4%} > {r_tol:.2%} in {rp.name}"}

            # Non-GIS scientific tabular + metadata.json physical conservation audit
            if not orig_layers and not orig_rasters and (src_dir / "metadata.json").exists():
                ok_tab, tab_reason, rep_changed = _audit_tabular_metadata_dir(src_dir, dst_dir, tol_rel)
                if not ok_tab:
                    return {"certified": False, "stage": "conservation", "reason": tab_reason}
                if rep_changed:
                    crs_changed = True
            audited_dirs += 1

        if max_disp_m < min_displacement_m and not crs_changed:
            return {
                "certified": False,
                "stage": "non_vacuity",
                "reason": f"Operator is vacuous/near-identity (max_disp={max_disp_m/1000:.1f} km < {min_displacement_m/1000:.0f} km, crs_changed={crs_changed})",
            }

        # Active discrimination: must expose >=1 failure/drift among currently passing scripts
        caught = 0
        caught_examples = []
        for c_idx, (wdir, code, orig_val, ans_type) in enumerate(passing_cases):
            pdir = tmp_path / f"disc_{c_idx}"
            try:
                probe_fn(wdir, pdir, 99 + c_idx)
            except Exception as exc:
                return {"certified": False, "stage": "disc_exec", "reason": f"perturb() failed on {wdir.name}: {exc}"}
            res = run_python_sandbox(code, pdir, timeout=timeout, python=python)
            if res["timed_out"] or res["returncode"] != 0:
                caught += 1
                tail = (res["stderr"].strip().splitlines() or ["timeout"])[-1][:140]
                caught_examples.append({"dir": wdir.name, "kind": "crash", "detail": tail})
                continue
            pval = parse_final_number(res["stdout"])
            if pval is None:
                caught += 1
                caught_examples.append({"dir": wdir.name, "kind": "no_number"})
                continue
            rel = abs(pval - orig_val) / max(abs(orig_val), 1e-12) if ans_type != "int" else (0.0 if round(pval) == round(orig_val) else 1.0)
            if rel > 0.01:
                caught += 1
                caught_examples.append({"dir": wdir.name, "kind": "drift", "orig": orig_val, "pert": pval})

        if passing_cases and caught == 0:
            return {
                "certified": False,
                "stage": "discrimination",
                "reason": f"Operator is geometrically valid (max_area_err={max_area_err:.5f}, disp={max_disp_m/1000:.0f}km) but exposed 0/{len(passing_cases)} failures on current passing scripts.",
            }

        return {
            "certified": True,
            "stage": "passed",
            "reason": f"Certified! audited={audited_dirs}, max_area_err={max_area_err:.5%}, max_len_err={max_len_err:.5%}, max_dist_err={max_dist_err:.5%}, max_disp_km={max_disp_m/1000:.1f}, caught={caught}/{len(passing_cases)}",
            "metrics": {
                "audited_dirs": audited_dirs,
                "max_area_err": max_area_err,
                "max_len_err": max_len_err,
                "max_dist_err": max_dist_err,
                "max_disp_km": round(max_disp_m / 1000.0, 1),
                "topo_mismatches": topo_mismatches,
                "caught": caught,
                "tested_scripts": len(passing_cases),
                "caught_examples": caught_examples[:5],
            },
        }


def _extract_hyp_and_code(text: str) -> tuple[str, str | None]:
    raw = _strip_thinking_preamble(text)
    first_fence = raw.find("```")
    prefix = raw[:first_fence] if first_fence != -1 else raw
    if "HYPOTHESIS:" in prefix:
        raw = raw[prefix.rfind("HYPOTHESIS:") :]
    hyp = ""
    m = re.search(r"HYPOTHESIS:\s*(.+?)(?=\n```|\Z)", raw, re.S)
    if m:
        hyp = m.group(1).strip()
    return hyp, _extract_fenced_blocks(raw, required_def="perturb")


def propose_and_certify_verifier(
    meta_llm: LLMClient,
    eval_result: dict,
    data_root: Path,
    *,
    existing_probe_summary: str = "Mid-latitude local AEQD relocation to (58.3N, 6.0E) and (-56.2S, 12.0E) with storage CRS flip.",
    max_attempts: int = 5,
    audit_sample_n: int = 10,
    disc_sample_n: int = 12,
    snippet_n: int = 6,
    pick_seed: int = 0,
) -> dict:
    """Co-evolve a new zero-label certified invariance probe from currently passing solver traces."""
    records = eval_result["records"]
    passing = [
        r for r in records
        if r.get("in_loop_pass") and r.get("code") and r.get("dev_low_value") is not None
    ]
    # Audit conservation across ALL unlabeled dev modalities (local, raster, large-extent),
    # while checking active discrimination strictly on currently passing solver scripts.
    strat_all = _stratified_pick(records, max(audit_sample_n, 15), pick_seed)
    strat_passing = _stratified_pick(passing, max(disc_sample_n, snippet_n), pick_seed)
    dev_dirs: list[Path] = []
    passing_cases: list[tuple[Path, str, float, str]] = []
    seen_fams = set()
    for r in strat_all + records:
        low_dir = data_root / r["task_id"] / "low"
        if not low_dir.exists():
            continue
        if r["family"] not in seen_fams and len(dev_dirs) < audit_sample_n:
            dev_dirs.append(low_dir)
            seen_fams.add(r["family"])
    for r in strat_passing + passing:
        low_dir = data_root / r["task_id"] / "low"
        if not low_dir.exists():
            continue
        case = (low_dir, r["code"], float(r["dev_low_value"]), r["answer"])
        if len(passing_cases) < disc_sample_n and all(c[0] != low_dir for c in passing_cases):
            passing_cases.append(case)

    snippet_recs = _stratified_pick(passing, snippet_n, pick_seed)
    snippets = []
    for i, r in enumerate(snippet_recs, 1):
        low_dir = data_root / r["task_id"] / "low"
        fnames = sorted(p.name for p in low_dir.iterdir() if p.is_file()) if low_dir.exists() else []
        full_code = r["code"] or ""
        if len(full_code) > 2400:
            code_disp = full_code[:2400] + f"\n# ... [display truncated; full script is {len(full_code)} chars]"
        else:
            code_disp = full_code
        meta_preview = ""
        meta_path = low_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta_txt = meta_path.read_text()[:500]
                meta_preview = f"\nMetadata preview (`metadata.json`):\n```json\n{meta_txt}\n```\n"
            except Exception:
                meta_preview = ""
        snippets.append(f"### Passing Script {i} (files: {', '.join(fnames)}){meta_preview}\n```python\n{code_disp}\n```")

    file_types = sorted({p.suffix.lower() for d in dev_dirs for p in d.iterdir() if p.is_file()})
    base_prompt = (
        f"## Existing Invariance Probes (T_k)\n{existing_probe_summary}\n\n"
        f"## Workspace File Types\n{', '.join(file_types)}\n\n"
        f"## Sample Scripts Passing T_k ({len(passing)}/{len(records)} dev tasks currently pass)\n"
        + "\n\n".join(snippets)
        + "\n\nInspect the passing scripts closely. Identify ONE concrete untested input or representation "
          "condition where these scripts rely on an implicit assumption that is NOT guaranteed on equivalent "
          "inputs, and write an invariant `perturb(src_dir, dst_dir, seed=42)` operator that exposes this "
          "fragility while passing all conservation audits."
    )

    attempts = []
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + (f"\n\n## Previous Attempt Audit Feedback\n{feedback}" if feedback else "")
        resp = meta_llm.complete(VERIFIER_SYSTEM_PROMPT, prompt)
        if resp.raw_error and not resp.text:
            feedback = f"LLM error: {resp.raw_error}"
            attempts.append({"attempt": attempt, "certified": False, "stage": "llm", "reason": feedback})
            continue
        hyp, code = _extract_hyp_and_code(resp.text)
        if not code:
            feedback = "Missing complete ```python ... ``` block defining perturb(src_dir, dst_dir, seed=42). Be concise and finish the full function inside a single fenced Python block."
            attempts.append({
                "attempt": attempt,
                "certified": False,
                "stage": "extract",
                "reason": feedback,
                "output_tokens": resp.output_tokens,
                "raw_tail": (resp.text or "")[-300:],
            })
            continue
        cert = certify_perturbation_operator(code, dev_dirs, passing_cases)
        entry = {"attempt": attempt, "hypothesis": hyp, "output_tokens": resp.output_tokens, **cert}
        if cert.get("stage") == "ast":
            entry["raw_tail"] = code[-300:]
        attempts.append(entry)
        if cert["certified"]:
            return {"success": True, "hypothesis": hyp, "code": code, "audit": cert, "attempts": attempts}
        feedback = f"Attempt {attempt} REJECTED at stage `{cert['stage']}`: {cert['reason']}. Fix this issue and ensure exact physical/geometric conservation + active failure exposure."

    return {"success": False, "hypothesis": "", "code": None, "audit": None, "attempts": attempts}

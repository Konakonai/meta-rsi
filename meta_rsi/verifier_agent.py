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
2. Allowed imports: stdlib (`pathlib`, `shutil`, `math`, `random`) plus `geopandas`, `pyproj`, `shapely`,
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
            try:
                probe_fn(src_dir, dst_dir, 42 + idx)
            except Exception as exc:
                return {"certified": False, "stage": "exec", "reason": f"perturb() crashed on {src_dir.name}: {type(exc).__name__}: {exc}"}

            orig_files = sorted(p.name for p in src_dir.iterdir() if p.is_file())
            pert_files = sorted(p.name for p in dst_dir.iterdir() if p.is_file()) if dst_dir.exists() else []
            if orig_files != pert_files:
                return {"certified": False, "stage": "schema", "reason": f"File list mismatch: {orig_files} vs {pert_files}"}

            orig_layers = _load_layers_for_audit(src_dir)
            pert_layers = _load_layers_for_audit(dst_dir)
            if set(orig_layers.keys()) != set(pert_layers.keys()):
                return {"certified": False, "stage": "schema", "reason": "Spatial layer keys mismatch"}

            for fname, (g_orig, crs_orig) in orig_layers.items():
                g_pert, crs_pert = pert_layers[fname]
                if len(g_orig) != len(g_pert):
                    return {"certified": False, "stage": "schema", "reason": f"Row count mismatch in {fname}"}
                cols_o = sorted(c for c in g_orig.columns if c != "geometry")
                cols_p = sorted(c for c in g_pert.columns if c != "geometry")
                if cols_o != cols_p:
                    return {"certified": False, "stage": "schema", "reason": f"Column mismatch in {fname}: {cols_o} vs {cols_p}"}
                # Attribute multiset preservation (row-order invariant)
                if cols_o:
                    ms_o = sorted(g_orig[cols_o].astype(str).agg("|".join, axis=1).tolist())
                    ms_p = sorted(g_pert[cols_o].astype(str).agg("|".join, axis=1).tolist())
                    if ms_o != ms_p:
                        return {"certified": False, "stage": "schema", "reason": f"Attribute multiset mutated in {fname}"}
                # Bounds sanity
                if g_pert.geometry.isna().any() or g_pert.geometry.is_empty.any():
                    return {"certified": False, "stage": "geometry", "reason": f"Empty/NaN geometry in {fname}"}
                tb = g_pert.total_bounds
                if not np.all(np.isfinite(tb)) or tb[0] < -180.0 or tb[2] > 180.0 or tb[1] < -90.0 or tb[3] > 90.0:
                    return {"certified": False, "stage": "geometry", "reason": f"Invalid WGS84 bounds {tb} in {fname}"}

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
                        return {"certified": False, "stage": "conservation", "reason": f"Geodesic area error {ae:.4%} > {tol_rel:.2%} in {fname}"}
                if l0 > 1.0:
                    le = abs(l1 - l0) / l0
                    max_len_err = max(max_len_err, le)
                    if le > tol_rel:
                        return {"certified": False, "stage": "conservation", "reason": f"Geodesic length error {le:.4%} > {tol_rel:.2%} in {fname}"}

                # Pairwise geodesic distance spectrum conservation
                d0 = _pairwise_geod_spectrum(g_orig)
                d1 = _pairwise_geod_spectrum(g_pert)
                if len(d0) > 0:
                    mask = d0 > 10.0
                    if np.any(mask):
                        de = float(np.max(np.abs(d1[mask] - d0[mask]) / d0[mask]))
                        max_dist_err = max(max_dist_err, de)
                        if de > max(tol_rel * 3.0, 3e-3):
                            return {"certified": False, "stage": "conservation", "reason": f"Pairwise geodesic dist error {de:.4%} in {fname}"}

            # Cross-layer topological conservation
            t0 = _topo_counts(orig_layers)
            t1 = _topo_counts(pert_layers)
            if t0 != t1:
                topo_mismatches += 1
                return {"certified": False, "stage": "topology", "reason": f"Topological counts changed on {src_dir.name}: {t0} vs {t1}"}

            # GeoTIFF raster conservation (when rasters are present)
            orig_rasters = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in _RASTER_EXTS)
            for rp in orig_rasters:
                pp = dst_dir / rp.name
                try:
                    rs0 = _raster_audit_stats(rp)
                    rs1 = _raster_audit_stats(pp)
                except Exception as exc:
                    return {"certified": False, "stage": "raster", "reason": f"Failed reading raster {rp.name}: {exc}"}
                if rs1["n_valid"] == 0 or not np.isfinite(rs1["mean"]):
                    return {"certified": False, "stage": "raster", "reason": f"Perturbed raster {rp.name} has no valid pixels"}
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
    raw = text or ""
    fence_m = re.search(r"^\s*```(?:python|py)?\s*$", raw, re.M)
    if fence_m:
        preamble = raw[: fence_m.start()]
        for end in ("</think>", "<channel|>"):
            if end in preamble:
                preamble = preamble.rsplit(end, 1)[1]
        raw = preamble + raw[fence_m.start() :]
    else:
        for end in ("</think>", "<channel|>"):
            if end in raw:
                raw = raw.rsplit(end, 1)[1]
    hyp = ""
    m = re.search(r"HYPOTHESIS:\s*(.+?)(?=\n```|\Z)", raw, re.S)
    if m:
        hyp = m.group(1).strip()
    lines = raw.splitlines()
    start_idx, end_idx = None, None
    for i, ln in enumerate(lines):
        if start_idx is None and re.match(r"^\s*```(?:python|py)?\s*$", ln):
            start_idx = i + 1
        elif start_idx is not None and re.match(r"^\s*```\s*$", ln):
            end_idx = i
    if start_idx is not None and end_idx is not None and end_idx > start_idx:
        return hyp, "\n".join(lines[start_idx:end_idx]).strip()
    return hyp, None


def propose_and_certify_verifier(
    meta_llm: LLMClient,
    eval_result: dict,
    data_root: Path,
    *,
    existing_probe_summary: str = "Mid-latitude local AEQD relocation to (58.3N, 6.0E) and (-56.2S, 12.0E) with GPKG CRS flip.",
    max_attempts: int = 4,
    audit_sample_n: int = 8,
    disc_sample_n: int = 8,
) -> dict:
    """Co-evolve a new zero-label certified invariance probe from currently passing solver traces."""
    records = eval_result["records"]
    passing = [
        r for r in records
        if r.get("in_loop_pass") and r.get("code") and r.get("dev_low_value") is not None
    ]
    # Collect unlabeled local workspace dirs and passing scripts
    dev_dirs: list[Path] = []
    passing_cases: list[tuple[Path, str, float, str]] = []
    seen_fams = set()
    for r in passing:
        # Infer low dir from task_id under data_root
        low_dir = data_root / r["task_id"] / "low"
        if not low_dir.exists():
            continue
        if r["family"] not in seen_fams and len(dev_dirs) < audit_sample_n:
            dev_dirs.append(low_dir)
            seen_fams.add(r["family"])
        if len(passing_cases) < disc_sample_n:
            passing_cases.append((low_dir, r["code"], float(r["dev_low_value"]), r["answer"]))

    snippets = []
    for i, r in enumerate(passing[:5], 1):
        snippets.append(f"### Passing Script {i}\n```python\n{(r['code'] or '')[:1000]}\n```")

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
            feedback = "Missing ```python ... ``` block defining perturb(src_dir, dst_dir, seed=42)."
            attempts.append({"attempt": attempt, "certified": False, "stage": "extract", "reason": feedback})
            continue
        cert = certify_perturbation_operator(code, dev_dirs, passing_cases)
        attempts.append({"attempt": attempt, "hypothesis": hyp, **cert})
        if cert["certified"]:
            return {"success": True, "hypothesis": hyp, "code": code, "audit": cert, "attempts": attempts}
        feedback = f"Attempt {attempt} REJECTED at stage `{cert['stage']}`: {cert['reason']}. Fix this issue and ensure exact geodesic conservation + active failure exposure."

    return {"success": False, "hypothesis": "", "code": None, "audit": None, "attempts": attempts}

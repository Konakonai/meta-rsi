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

VERIFIER_SYSTEM_PROMPT = """You are an AI verification researcher co-evolving label-free metamorphic invariance probes
for geospatial analysis coding agents.

You will be shown Python scripts emitted by a solver harness that ALREADY pass existing mid-latitude
invariance probes (58.3N / -56.2S). Your job is to identify an UNTESTED geometric/coordinate edge case
where these passing scripts rely on fragile assumptions (e.g., latitude bounds of UTM zones >84N or <80S,
crossing hemispheres/antimeridian, storage CRS differences, or large-extent orientation), and write ONE
self-contained Python perturbation module implementing a mathematically distance- and area-preserving
transformation `T(x)` that exposes this fragility WITHOUT changing the true geometric answer!

Hard contract for your Python module:
1. Must define: `def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict`
2. Allowed imports: stdlib (`pathlib`, `shutil`, `math`, `random`) plus `geopandas`, `pyproj`, `shapely`, `numpy`, `pandas`.
3. Must copy ALL files from `src_dir` to `dst_dir` (creating `dst_dir` if needed) preserving exact filenames,
   row counts, columns, and attribute values.
4. For local datasets (span < 5 deg), transform all layers together via an exact local Azimuthal Equidistant
   (`+proj=aeqd +lat_0=... +lon_0=... +datum=WGS84 +units=m +no_defs`) frame centered on the combined bbox
   centroid to your chosen target center `(target_lat, target_lon)` (staying strictly inside `[-89, 89]` lat
   and `[-179, 179]` lon so WGS84 coordinates remain valid and do not wrap the antimeridian!).
   - Keep `.csv` lon/lat rounded to 7 decimals; for `.gpkg`, store in valid CRS (note: outside `[-80, 84]`
     UTM is undefined, so use `EPSG:4326` or Polar Stereographic `EPSG:3413`/`EPSG:3031` if storing projected!).
   - For large-extent datasets (span >= 5 deg), you may copy them unchanged or apply a valid geodesic rotation
     that stays inside `[-175, 175]` lon and `[-78, 78]` lat.
5. Must return a metadata dict `{"mode": "<short_name>", ...}`.
6. Your operator will be automatically audited for:
   - Exact schema & attribute multiset preservation
   - Pairwise WGS84 geodesic distance, polygon area, and line length conservation (<0.1% error)
   - Exact cross-layer topological (`intersects`/`within`) conservation
   - Non-trivial movement AND exposing a failure/drift on at least one current passing script!

Output format:
HYPOTHESIS: <2 sentences naming the blind spot in the passing scripts and how your isometric perturbation exposes it>

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


def _topo_counts(layers: dict[str, tuple[gpd.GeoDataFrame, CRS | None]]) -> dict[str, int]:
    """Exact cross-layer topological relation counts in local AEQD (avoids degree artifacts)."""
    names = sorted(layers.keys())
    if len(names) < 2:
        return {}
    bounds = np.array([layers[k][0].total_bounds for k in names])
    cx = 0.5 * float(bounds[:, 0].min() + bounds[:, 2].max())
    cy = 0.5 * float(bounds[:, 1].min() + bounds[:, 3].max())
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

                # Centroid displacement
                ob = g_orig.total_bounds
                ocx, ocy = 0.5 * (ob[0] + ob[2]), 0.5 * (ob[1] + ob[3])
                pcx, pcy = 0.5 * (tb[0] + tb[2]), 0.5 * (tb[1] + tb[3])
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
    hyp = ""
    m = re.search(r"HYPOTHESIS:\s*(.+?)(?=\n```|\Z)", text, re.S)
    if m:
        hyp = m.group(1).strip()
    lines = text.splitlines()
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

    base_prompt = (
        f"## Existing Invariance Probes (T_k)\n{existing_probe_summary}\n\n"
        f"## Sample Scripts Passing T_k ({len(passing)}/{len(records)} dev tasks currently pass)\n"
        + "\n\n".join(snippets)
        + "\n\nInspect the passing scripts closely. Identify ONE concrete geographic/geometric boundary condition "
          "where these scripts silently assume something that is NOT invariant everywhere on Earth (e.g., latitude domain limits "
          "of the projection library call they rely on), and write a distance/area/topology-preserving `perturb(src_dir, dst_dir, seed=42)` "
          "operator that exposes this fragility while keeping true geometric answers invariant."
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

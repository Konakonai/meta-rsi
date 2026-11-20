"""Non-evolution inference-time metamorphic self-check baseline (`runtime_meta`).

Starts from minimal workspace inspection (NO explicit CRS/geodesic prompt rules!),
runs the generated script on `work_dir`, and then performs an in-harness label-free
metamorphic self-consistency check inside `solve()` using `run_python(code, pert_dir)`.
If the script's output drifts >1% or crashes under equivalent relocation, the harness
feeds the drift feedback back to the Task LLM for runtime self-repair.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer

SYSTEM_PROMPT = (
    "You are a Python geospatial data analysis assistant. "
    "Write one self-contained Python script inside a ```python ... ``` block that reads files in "
    "the current working directory and prints ONLY the final numerical answer on the last line. "
    "Available libraries: geopandas, pyproj, shapely, pandas, numpy, scipy."
)


def _inspect_workspace(work_dir: Path) -> str:
    lines = ["Workspace files:"]
    for p in sorted(work_dir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext in (".geojson", ".gpkg"):
                g = gpd.read_file(p)
                cols = [c for c in g.columns if c != "geometry"]
                lines.append(f"- {p.name}: rows={len(g)}, crs={g.crs}, geom={list(g.geom_type.unique())}, cols={cols}")
            elif ext == ".csv":
                df = pd.read_csv(p)
                lines.append(f"- {p.name}: rows={len(df)}, cols={list(df.columns)}")
        except Exception as exc:
            lines.append(f"- {p.name}: inspect_error={exc}")
    return "\n".join(lines)


def _extract_code(text: str) -> str | None:
    final = text or ""
    for end in ("</think>", "<channel|>"):
        if end in final:
            final = final.rsplit(end, 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", final, re.S)
    return blocks[-1].strip() if blocks else None


def _parse_number(stdout: str) -> float | None:
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        v = float(lines[-1].replace(",", ""))
        return None if (v != v or abs(v) == float("inf")) else v
    except ValueError:
        return None


def _make_relocated_copy(src_dir: Path, dst_dir: Path, target_lat: float = 58.3, target_lon: float = 6.0) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in (".geojson", ".gpkg"):
            g = gpd.read_file(p)
            g4326 = g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")
            layers.append((p, ext, g4326, g.crs))
        elif ext == ".csv":
            df = pd.read_csv(p)
            if {"lon", "lat"}.issubset(df.columns):
                g = gpd.GeoDataFrame(df.drop(columns=["lon", "lat"]), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
                layers.append((p, ".csv", g, CRS.from_epsg(4326)))
            else:
                shutil.copy2(p, dst_dir / p.name)
        else:
            shutil.copy2(p, dst_dir / p.name)
    if not layers:
        return False
    bounds = np.array([g.total_bounds for _, _, g, _ in layers])
    minx, miny = float(bounds[:, 0].min()), float(bounds[:, 1].min())
    maxx, maxy = float(bounds[:, 2].max()), float(bounds[:, 3].max())
    if max(maxx - minx, maxy - miny) >= 5.0:
        return False
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    c1 = CRS.from_proj4(f"+proj=aeqd +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m +no_defs")
    c2 = CRS.from_proj4(f"+proj=aeqd +lat_0={target_lat} +lon_0={target_lon} +datum=WGS84 +units=m +no_defs")
    t1 = Transformer.from_crs("EPSG:4326", c1, always_xy=True)
    t2 = Transformer.from_crs(c2, "EPSG:4326", always_xy=True)

    def _xform(geoms, tr):
        def f(c):
            x, y = tr.transform(c[:, 0], c[:, 1])
            return np.column_stack([x, y])
        return [shapely.transform(g, f) for g in geoms]

    for p, ext, g4326, orig_crs in layers:
        g_new = gpd.GeoDataFrame(
            g4326.drop(columns="geometry").copy(),
            geometry=_xform(_xform(list(g4326.geometry), t1), t2),
            crs="EPSG:4326",
        )
        out = dst_dir / p.name
        if ext == ".csv":
            df = pd.DataFrame(g_new.drop(columns="geometry"))
            df["lon"] = g_new.geometry.x.round(7)
            df["lat"] = g_new.geometry.y.round(7)
            df.to_csv(out, index=False)
        elif ext == ".gpkg":
            if orig_crs is not None and not orig_crs.is_geographic:
                g_new = g_new.to_crs(g_new.estimate_utm_crs())
            g_new.to_file(out, driver="GPKG")
        else:
            g_new.to_file(out, driver="GeoJSON")
    return True


def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict:
    inspect = _inspect_workspace(work_dir)
    base_user = f"{inspect}\n\nTask:\n{task_prompt}"
    trace = []
    code = None
    user_msg = base_user

    for turn in range(4):
        reply = llm_call(SYSTEM_PROMPT, user_msg)
        code = _extract_code(reply)
        if not code:
            trace.append({"turn": turn, "status": "no_code"})
            user_msg = f"{base_user}\n\nReturn a valid ```python``` block."
            continue
        res = run_python(code, work_dir)
        val = _parse_number(res["stdout"]) if res["returncode"] == 0 and not res["timed_out"] else None
        if res["returncode"] != 0 or res["timed_out"] or val is None:
            trace.append({"turn": turn, "status": "exec_err"})
            err = "TIMED OUT" if res["timed_out"] else (res["stderr"][-500:] or res["stdout"][-200:])
            user_msg = f"{base_user}\n\nPrevious script failed:\n```python\n{code}\n```\nError:\n{err}\nFix and return complete code."
            continue

        # In-harness runtime self-consistency check on relocated workspace
        if turn < 3:
            with tempfile.TemporaryDirectory() as tmp:
                pdir = Path(tmp) / "pert"
                if _make_relocated_copy(work_dir, pdir):
                    pres = run_python(code, pdir)
                    pval = _parse_number(pres["stdout"]) if pres["returncode"] == 0 and not pres["timed_out"] else None
                    if pval is None:
                        trace.append({"turn": turn, "status": "meta_crash", "val": val})
                        perr = pres["stderr"][-400:] if pres["returncode"] != 0 else "no numeric output"
                        user_msg = (
                            f"{base_user}\n\nYour script printed {val:.6g}, but FAILED when tested on an equivalent "
                            f"geometrically relocated dataset: {perr}\n"
                            f"Previous script:\n```python\n{code}\n```\n"
                            "Fix the script so it works reliably at any location and CRS without hardcoding zones."
                        )
                        continue
                    rel_diff = abs(pval - val) / max(abs(val), 1e-12)
                    if rel_diff > 0.01:
                        ratio = pval / val if abs(val) > 1e-12 else float("inf")
                        trace.append({"turn": turn, "status": "meta_drift", "val": val, "pval": pval})
                        user_msg = (
                            f"{base_user}\n\nYour script printed {val:.6g}, but when run on an equivalent relocated copy "
                            f"with identical local geometry it printed {pval:.6g} (ratio={ratio:.3g}x, >1% drift!).\n"
                            f"Previous script:\n```python\n{code}\n```\n"
                            "Your metric calculation is location/projection-dependent. Fix the projection/measurement method "
                            "so results are geometrically invariant everywhere on Earth."
                        )
                        continue
        trace.append({"turn": turn, "status": "ok", "val": val})
        return {"code": code, "trace": trace, "turns": turn + 1}

    return {"code": code, "trace": trace, "turns": len(trace)}

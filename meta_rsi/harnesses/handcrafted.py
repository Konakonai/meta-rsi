"""Non-evolution strong baseline: 15-minute GIS engineer hand-written Agent Harness.

Combines clean workspace inspection (files, CRS, extent, columns, installed packages),
concise geodetic/CRS best-practice rules, and a 3-turn execute-inspect-repair loop.
Serves as a realistic engineering anchor to quantify the net value of RSI evolution.
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

SYSTEM_PROMPT = """You are an expert geospatial Python analyst.
Write one self-contained Python script inside a single ```python ... ``` block that reads the files in the current working directory and prints ONLY the final numerical answer on the last line.

Core rules:
1. Available libraries: geopandas, pyproj, shapely, pandas, numpy, scipy (do NOT import geopy or networkx).
2. Always inspect/respect each file's CRS. CSV files with lon/lat are EPSG:4326.
3. Never use EPSG:3857 (Web Mercator) or raw degrees for metric distances, lengths, areas, or buffers.
   - If spatial extent < 5 degrees: project all layers to ONE shared local metric CRS:
     use `crs = gdf.to_crs("EPSG:4326").estimate_utm_crs()` when |lat| < 80, or a local Azimuthal Equidistant
     `f"+proj=aeqd +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m +no_defs"` centered on the data bbox (works globally including poles).
   - If spatial extent >= 5 degrees: use `pyproj.Geod(ellps="WGS84")` (`geometry_area_perimeter` / `geometry_length` / `line_length`) on EPSG:4326 geometries (or `EPSG:6933` for equal-area).
4. After `gpd.sjoin` / `gpd.overlay`, watch for renamed duplicate columns (`id_left`, `id_right`) and deduplicate when counting covered features.
5. Never wrap everything in bare `try/except` that prints 0 on error.
"""


def _inspect_workspace(work_dir: Path) -> str:
    lines = ["Workspace files:"]
    for p in sorted(work_dir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext in (".geojson", ".gpkg"):
                g = gpd.read_file(p)
                g4326 = g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")
                b = g4326.total_bounds
                span = max(b[2] - b[0], b[3] - b[1])
                cols = [c for c in g.columns if c != "geometry"]
                lines.append(
                    f"- {p.name}: rows={len(g)}, crs={g.crs}, geom={list(g.geom_type.unique())}, "
                    f"cols={cols}, wgs84_bounds={[round(float(x), 3) for x in b]}, span_deg={span:.3f}"
                )
            elif ext == ".csv":
                df = pd.read_csv(p)
                info = f"- {p.name}: rows={len(df)}, cols={list(df.columns)}"
                if {"lon", "lat"}.issubset(df.columns) and len(df) > 0:
                    span = max(float(df["lon"].max() - df["lon"].min()), float(df["lat"].max() - df["lat"].min()))
                    info += f", lon=[{df['lon'].min():.3f},{df['lon'].max():.3f}], lat=[{df['lat'].min():.3f},{df['lat'].max():.3f}], span_deg={span:.3f}"
                lines.append(info)
            else:
                lines.append(f"- {p.name}")
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
            user_msg = f"{base_user}\n\nYour previous reply had no ```python``` block. Return a complete Python script."
            continue
        res = run_python(code, work_dir)
        val = _parse_number(res["stdout"]) if res["returncode"] == 0 and not res["timed_out"] else None
        trace.append({"turn": turn, "rc": res["returncode"], "timed_out": res["timed_out"], "val": val})
        if res["returncode"] == 0 and not res["timed_out"] and val is not None:
            return {"code": code, "trace": trace, "turns": turn + 1}
        err = "TIMED OUT" if res["timed_out"] else (res["stderr"][-600:] or f"stdout={res['stdout'][-200:]}")
        user_msg = (
            f"{base_user}\n\nPrevious script failed:\n```python\n{code}\n```\n"
            f"Error/Output:\n{err}\n\nFix the bug and return a complete corrected Python script."
        )
    return {"code": code, "trace": trace, "turns": len(trace)}

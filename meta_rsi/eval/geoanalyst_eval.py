"""External Zero-Shot Transfer Evaluation on GeoAnalystBench (Zhang et al., 2025, Trans. in GIS).

Evaluates generated Python spatial-analysis scripts across the 17 open-source CRS/metric/raster
tasks from `GeoDS/GeoAnalystBench/dataset/GeoAnalystBench.csv` using strictly deterministic AST
and geospatial behavioral checks (zero LLM-as-a-judge bias):
  - Vector metric tasks (`{8, 9, 10, 20, 31, 34, 44}`): tasks requiring metric buffer/area/length.
  - Raster tasks (`{3, 7, 11, 12, 20, 21, 22, 35}`): tasks using `rasterio` rasters.
  - Multi-layer CRS alignment tasks (`{1, 8, 9, 10, 11, 20, 34, 42, 44}`).

Reports granular, auditable rates:
  1. `syntax_ok`: code parses as valid Python AST.
  2. `uses_3857`: uses distorted Web Mercator (`EPSG:3857`) on metric operations (lower is better).
  3. `projected_metric`: explicitly projects to a non-3857 metric CRS or uses `pyproj.Geod` before
     metric ops (`.area`, `.length`, `.buffer`, `.distance`, `sjoin_nearest`).
  4. `dynamic_metric_crs`: uses data-driven projection (`estimate_utm_crs`, `+proj=aeqd/laea/aea`,
     or `pyproj.Geod`) rather than hardcoded regional EPSG codes.
  5. `raster_nodata_safe`: explicitly handles nodata/NaN masking on raster tasks.
  6. `crs_aligned`: aligns layer CRSs explicitly (`to_crs` / `reproject`).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..llm import LLMClient
from ..meta_agent import _extract_fenced_blocks, _strip_thinking_preamble

GEOANALYST_VECTOR_METRIC_IDS = {8, 9, 10, 20, 31, 34, 44}
GEOANALYST_RASTER_IDS = {3, 7, 11, 12, 20, 21, 22, 35}
GEOANALYST_ALIGN_IDS = {1, 8, 9, 10, 11, 20, 34, 42, 44}
GEOANALYST_METRIC_IDS = GEOANALYST_VECTOR_METRIC_IDS | GEOANALYST_RASTER_IDS | GEOANALYST_ALIGN_IDS | {24}

GEOANALYST_SYSTEM_PROMPT = """You are a senior geospatial Python engineer.
Write a complete, self-contained Python script to solve the user's spatial analysis task using open-source Python GIS libraries (`geopandas`, `pyproj`, `shapely`, `rasterio`, `numpy`, `pandas`, `matplotlib`, etc.).
Return your complete executable solution inside a single ```python ... ``` code block."""


def load_geoanalyst_tasks(csv_path: Path, metric_only: bool = True) -> list[dict]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    out = []
    for r in rows:
        if r.get("Open Source", "").strip().upper() != "T":
            continue
        tid = int(r["id"])
        if metric_only and tid not in GEOANALYST_METRIC_IDS:
            continue
        out.append({
            "id": tid,
            "task": r.get("Task", "").strip(),
            "instruction": r.get("Instruction", "").strip(),
            "domain_knowledge": r.get("Domain Knowledge", "").strip(),
            "dataset_description": r.get("Dataset Description", "").strip(),
            "needs_raster": tid in GEOANALYST_RASTER_IDS,
            "needs_vector_metric": tid in GEOANALYST_VECTOR_METRIC_IDS,
            "needs_crs_align": tid in GEOANALYST_ALIGN_IDS,
        })
    return out


def build_prompt(task: dict) -> str:
    parts = [f"## Task\n{task['task']}\n\n## Instruction\n{task['instruction']}"]
    if task["dataset_description"]:
        parts.append(f"## Dataset Description\n{task['dataset_description']}")
    if task["domain_knowledge"]:
        parts.append(f"## Domain Knowledge\n{task['domain_knowledge']}")
    return "\n\n".join(parts)


def extract_python_code(text: str) -> str | None:
    raw = _strip_thinking_preamble(text or "")
    blk = _extract_fenced_blocks(raw)
    if blk:
        return blk
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.S)
    return m[-1].strip() if m else None


def audit_geo_script(code: str | None, task: dict) -> dict:
    """Deterministic AST & geospatial behavioral audit for a GeoAnalystBench script."""
    if not code or not code.strip():
        return {
            "syntax_ok": False,
            "uses_3857": False,
            "no_3857_metric": False,
            "uses_metric_op": False,
            "uses_metric_proj": False,
            "dynamic_metric_crs": False,
            "no_degree_metric": False,
            "raster_nodata_safe": False,
            "crs_aligned": False,
            "strict_geo_sound": False,
            "failure_tags": ["no_code"],
        }

    try:
        tree = ast.parse(code)
        syntax_ok = True
    except SyntaxError:
        tree = None
        syntax_ok = False

    attr_names: set[str] = set()
    call_names: set[str] = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr_names.add(node.attr)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    call_names.add(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    call_names.add(node.func.id)

    code_nc = "\n".join(ln.split("#", 1)[0] for ln in code.splitlines())
    uses_3857 = bool(re.search(r"\b3857\b|epsg:3857", code_nc, re.I))
    uses_metric_op = bool(
        {"area", "length"} & attr_names
        or {"buffer", "distance", "sjoin_nearest"} & call_names
    )
    has_to_crs = "to_crs" in call_names
    dynamic_metric_crs = bool(
        "estimate_utm_crs" in call_names
        or "Geod" in call_names
        or re.search(r"\+proj=(?:aeqd|laea|aea|utm|tmerc|lcc)\b", code_nc, re.I)
    )
    # Match any non-4326, non-3857 projection target (named EPSG, dynamic UTM/AEQD/LAEA, or another layer's `.crs`)
    only_4326_to_crs = has_to_crs and not dynamic_metric_crs and not bool(
        re.search(
            r"to_crs\(\s*(?:epsg\s*=\s*)?(?!['\"]?(?:epsg:)?(?:4326|3857)['\"]?\s*\))",
            code_nc,
            re.I,
        )
    )
    uses_metric_proj = bool(dynamic_metric_crs or (has_to_crs and not only_4326_to_crs))

    no_3857_metric = not uses_3857
    if task["needs_vector_metric"]:
        no_degree_metric = uses_metric_proj and not uses_3857
    else:
        no_degree_metric = not uses_3857

    if task["needs_raster"]:
        raster_nodata_safe = bool(
            re.search(r"masked\s*=\s*True|\.nodata\b|np\.isnan|np\.ma\.|nodata\s*is\s*not\s*None|np\.isfinite|np\.where", code_nc)
        )
    else:
        raster_nodata_safe = True

    crs_aligned = bool(has_to_crs or ("reproject" in call_names) or (not task.get("needs_crs_align", False)))
    strict_geo_sound = bool(syntax_ok and no_3857_metric and no_degree_metric and raster_nodata_safe)

    tags = []
    if not syntax_ok:
        tags.append("syntax_error")
    if uses_3857:
        tags.append("web_mercator_3857")
    if not no_degree_metric:
        tags.append("unprojected_metric")
    if not raster_nodata_safe:
        tags.append("missing_raster_nodata_mask")

    return {
        "syntax_ok": syntax_ok,
        "uses_3857": uses_3857,
        "no_3857_metric": no_3857_metric,
        "uses_metric_op": uses_metric_op,
        "uses_metric_proj": uses_metric_proj,
        "dynamic_metric_crs": dynamic_metric_crs,
        "no_degree_metric": no_degree_metric,
        "raster_nodata_safe": raster_nodata_safe,
        "crs_aligned": crs_aligned,
        "strict_geo_sound": strict_geo_sound,
        "failure_tags": tags,
    }


def evaluate_geoanalyst(
    csv_path: Path,
    llm: LLMClient,
    *,
    system_prompt: str = GEOANALYST_SYSTEM_PROMPT,
    metric_only: bool = True,
    workers: int = 8,
) -> dict:
    tasks = load_geoanalyst_tasks(csv_path, metric_only=metric_only)
    t0 = time.time()

    def _one(task: dict) -> dict:
        prompt = build_prompt(task)
        resp = llm.complete(system_prompt, prompt)
        code = extract_python_code(resp.text)
        audit = audit_geo_script(code, task)
        return {
            "id": task["id"],
            "task": task["task"],
            "needs_raster": task["needs_raster"],
            "needs_vector_metric": task["needs_vector_metric"],
            "needs_crs_align": task["needs_crs_align"],
            "audit": audit,
            "code_len": len(code or ""),
            "code": code or "",
            "raw_error": resp.raw_error,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(_one, tasks))

    vec_recs = [r for r in records if r["needs_vector_metric"]]
    rast_recs = [r for r in records if r["needs_raster"]]
    align_recs = [r for r in records if r["needs_crs_align"]]
    n = max(len(records), 1)
    summary = {
        "n": len(records),
        "n_vec_metric": len(vec_recs),
        "n_raster": len(rast_recs),
        "n_align": len(align_recs),
        "syntax_ok": round(sum(1 for r in records if r["audit"]["syntax_ok"]) / n, 4),
        "uses_3857_all": round(sum(1 for r in records if r["audit"]["uses_3857"]) / n, 4),
        "uses_3857_vec": round(sum(1 for r in vec_recs if r["audit"]["uses_3857"]) / max(len(vec_recs), 1), 4),
        "projected_metric_vec": round(sum(1 for r in vec_recs if r["audit"]["no_degree_metric"]) / max(len(vec_recs), 1), 4),
        "dynamic_crs_vec": round(sum(1 for r in vec_recs if r["audit"]["dynamic_metric_crs"]) / max(len(vec_recs), 1), 4),
        "raster_nodata_safe": round(sum(1 for r in rast_recs if r["audit"]["raster_nodata_safe"]) / max(len(rast_recs), 1), 4),
        "crs_aligned": round(sum(1 for r in align_recs if r["audit"]["crs_aligned"]) / max(len(align_recs), 1), 4),
        "strict_geo_sound": round(sum(1 for r in records if r["audit"]["strict_geo_sound"]) / n, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "usage": llm.usage.to_dict(),
    }
    return {"summary": summary, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LLM on GeoAnalystBench CRS/metric Python tasks")
    ap.add_argument("--csv", default="data/geoanalyst/GeoAnalystBench.csv")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-open-source", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    llm = LLMClient("openai-compat", a.model, base_url=a.url, temperature=0.0, max_tokens=2048, enable_thinking=False)
    res = evaluate_geoanalyst(Path(a.csv), llm, metric_only=not a.all_open_source, workers=a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

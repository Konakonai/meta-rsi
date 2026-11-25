"""Run a model's script on all three location variants and classify the outcome.

Categories:
  correct       right on every variant
  error         crashed, timed out, or printed no number on some variant
  silent_wrong  ran everywhere, wrong everywhere
  lucky         ran everywhere, right on some variants only (e.g. EPSG:3857 near the equator)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REL_TOL = 0.01
# Answers of type "coordinate" are WGS84 latitudes/longitudes in decimal degrees. A 1% relative band
# would be ~0.6 deg (~67 km) at 60N and accept any latitude inside a city, so they get an absolute band:
# centroid-based answers printed to >= 4 decimals pass, bounding-box maxima (>= 1.1e-4 deg off on
# data/osm_global) and city-centre guesses (>= 2.9e-3 deg off) fail.
COORD_ABS_TOL_DEG = 5e-5


def extract_code(completion: str) -> str | None:
    final = completion
    for end in ("</think>", "<channel|>"):  # drop a thinking span when its end marker survives decoding
        if end in final:
            final = final.rsplit(end, 1)[1]
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", final, re.S)
    if not blocks:
        return None
    full = [b for b in blocks if "read_file" in b or "read_csv" in b]  # prefer the full script over snippets
    return (full or blocks)[-1]


def matches(value: float, truth: float, answer: str, rel_tol: float = REL_TOL) -> bool:
    if answer == "count":
        return abs(value - truth) < 0.5
    if answer == "coordinate":
        return abs(value - truth) <= COORD_ABS_TOL_DEG + 1e-9  # slack for float error at the rounding edge
    return abs(value - truth) <= rel_tol * max(abs(truth), 1e-12)


def run_variant(code: str, data_dir: Path, timeout: float = 30.0, python: str = sys.executable) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(data_dir, work)  # the script may write files; keep the dataset clean
        (work / "solution.py").write_text(code)
        try:
            r = subprocess.run([python, "solution.py"], cwd=work, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or [""])[-1][:200]
        return {"status": "exception", "stderr": tail}
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    try:  # the prompt asks for only the final number, so the last line must parse as one
        return {"status": "ok", "value": float(lines[-1].replace(",", ""))}
    except (IndexError, ValueError):
        return {"status": "no_number", "stdout": r.stdout[-200:]}


def score(task: dict, code: str | None, root: Path, timeout: float = 30.0, python: str = sys.executable) -> dict:
    if code is None:
        return {"category": "no_code", "reward": 0.0, "variants": {}}
    res = {}
    for v, info in task["variants"].items():
        r = run_variant(code, root / info["dir"], timeout, python)
        if r["status"] == "ok":
            r["correct"] = matches(r["value"], info["truth"], task["answer"])
        res[v] = r
    if any(r["status"] != "ok" for r in res.values()):
        cat = "error"
    else:
        n_ok = sum(r["correct"] for r in res.values())
        cat = "correct" if n_ok == len(res) else ("silent_wrong" if n_ok == 0 else "lucky")
    return {"category": cat, "reward": float(cat == "correct"), "variants": res}

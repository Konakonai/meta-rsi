"""Classify wrong answers with the Gemini Decisions API (gemini-decision-1).

One request per answer. The input is the task prompt, the script, and what the script printed on
each variant next to the truth; one `choice` question asks for the main cause. Run it on every
silent_wrong / lucky answer: on answers the rules already labelled it gives an agreement check,
on the rest (other, constant_ratio, zero_result, nan_or_inf) it gives the label.

    GEMINI_API_KEY=... python scripts/classify_llm.py --root data/v1 --rules results/errors.json \
        results/*.jsonl --out results/errors_llm.jsonl

Resumable: answers already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.gsb.harness import extract_code  # noqa: E402

URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
OPTIONS = {
    "web_mercator": "Measures distance, length, area or buffers in EPSG:3857 (Web Mercator), which is distorted away from the equator.",
    "degrees": "Measures distance, length, area or buffers directly on longitude/latitude degrees without projecting to metres.",
    "fixed_utm": "Uses a hard-coded UTM zone, or computes the UTM zone by hand with a bug, so the zone does not fit the data.",
    "one_zone_large": "Uses one UTM or other local projection for data spanning tens of degrees, where geodesic or equal-area methods are needed.",
    "crs_mismatch": "Combines layers or geometries in different CRSs without reprojecting them to a common CRS (e.g. reprojects one layer to EPSG:3857 or UTM but tests it against another layer or a point still in degrees).",
    "assumes_wgs84": "Treats projected coordinates as longitude/latitude (e.g. applies haversine or Geod, or set_crs(4326), to UTM data).",
    "latlon_order": "Passes coordinates in the wrong order to a distance function: (lon, lat) to geopy geodesic/great_circle or to a haversine that expects (lat, lon), or (lat, lon) to pyproj Geod.inv.",
    "nn_in_degrees": "Finds nearest neighbours (KD-tree, sindex, min of planar distance) on raw longitude/latitude and then measures the chosen pair correctly; away from the equator the chosen neighbour is often not the nearest.",
    "predicate": "Uses the wrong spatial predicate or join direction (within vs intersects vs contains, swapped sjoin sides, clip vs select).",
    "method_trap": "Aggregation trap: counts features covered by several buffers more than once, does not scale by the clipped area share, drops zero-count groups, or does not merge overlaps.",
    "misread": "Misreads the question: wrong parameter value, filter, column, feature, or aggregation (sum vs mean, max vs min).",
    "unit": "Converts units wrongly (m vs km, m2 vs km2 vs hectares).",
    "programming_bug": "A general programming mistake unrelated to geography: pandas/GeoSeries index misalignment (e.g. gdf.within(other_series) compares row by row), wrong tuple element from a library call (e.g. Geod.inv()[0] is an azimuth, not a distance), unpacking or off-by-one errors, an exception swallowed so 0 is printed, invented or hard-coded data instead of reading the files, string vs integer comparison.",
    "other": "None of the above.",
}
UNLABELLED = {"other", "constant_ratio", "zero_result", "nan_or_inf"}
QUESTION = {
    "cause": {
        "type": "choice",
        "instructions": ("The Python script below was written to answer the GIS task, and it ran without errors, but "
                         "its printed number is wrong on at least one of three datasets that have the same local "
                         "geometry at different places on the Earth. Using the code and the printed values versus "
                         "the true values, what is the main cause of the wrong result? If several mistakes are present, choose the one that "
                         "makes the printed numbers wrong."),
        "criteria": OPTIONS,
    }
}


def build_input(task, row, code):
    lines = [f"TASK\n{task['prompt']}\n"]
    if task["store_utm"]:
        lines.append("DATA NOTE: the GeoPackage layers in this task are stored in a local UTM CRS; GeoJSON and CSV "
                     "layers are WGS84 longitude/latitude.\n")
    else:
        lines.append("DATA NOTE: all layers are WGS84 longitude/latitude.\n")
    lines.append(f"SCRIPT\n```python\n{code}\n```\n")
    lines.append("RESULTS (printed value vs true value)")
    for v, info in task["variants"].items():
        got = row["variants"][v]["value"]
        lines.append(f"- dataset centred at lat {info['lat']:.1f}, lon {info['lon']:.1f}: printed {got:.6g}, "
                     f"true {info['truth']:.6g}")
    return "\n".join(lines)


def call(key, text, retries=6):
    body = json.dumps({"model": "gemini-decision-1", "input": text, "questions": QUESTION}).encode()
    for i in range(retries):
        req = urllib.request.Request(URL, body, {"x-goog-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 503) and i < retries - 1:
                time.sleep(2 ** i)
                continue
            return {"error": {"http": e.code, "message": msg}}
        except (urllib.error.URLError, TimeoutError) as e:
            if i < retries - 1:
                time.sleep(2 ** i)
                continue
            return {"error": {"message": str(e)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/v1")
    ap.add_argument("--rules", required=True, help="output of classify_errors.py")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--unlabelled-only", action="store_true",
                    help="only answers the rules could not place (other, constant_ratio, zero_result, nan_or_inf)")
    a = ap.parse_args()
    key = os.environ["GEMINI_API_KEY"]
    tasks = {t["task_id"]: t for t in map(json.loads, open(Path(a.root) / "tasks.jsonl")) if t["split"] == "test"}
    rules = json.load(open(a.rules))
    done = set()
    if Path(a.out).exists():
        done = {(r["run"], r["task_id"], r["sample"]) for r in map(json.loads, open(a.out)) if "error" not in r}
    jobs = []
    for f in a.files:
        run = Path(f).stem
        lab = {(l["task_id"], l["sample"]): l["label"] for l in rules[run]}
        for r in map(json.loads, open(f)):
            k = (r["task_id"], r["sample"])
            if a.unlabelled_only and k in lab and lab[k] not in UNLABELLED:
                continue
            if k in lab and (run, *k) not in done:
                jobs.append((run, r, lab[k]))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} answers to classify", flush=True)

    def one(job):
        run, r, rule = job
        t = tasks[r["task_id"]]
        res = call(key, build_input(t, r, extract_code(r["text"]) or ""))
        out = {"run": run, "task_id": r["task_id"], "sample": r["sample"], "family": r["family"], "rule": rule}
        if "error" in res:
            out["error"] = res["error"]
        else:
            ans = res["answers"]["cause"]
            out.update(llm=ans["choice"], confidence=ans["confidence"], probabilities=ans["probabilities"],
                       tokens=res.get("usage", {}).get("total_input_tokens"))
        return out

    n_err = 0
    with open(a.out, "a") as fo, ThreadPoolExecutor(a.workers) as ex:
        for i, out in enumerate(ex.map(one, jobs), 1):
            n_err += "error" in out
            fo.write(json.dumps(out) + "\n")
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {n_err} errors", flush=True)
            if "error" in out and n_err <= 3:
                print("ERROR", out["error"], flush=True)


if __name__ == "__main__":
    main()

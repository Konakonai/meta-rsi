"""Rule-based classification of silent_wrong and lucky answers.

Each wrong answer is matched against answers produced by known wrong methods, re-solved on the
same three datasets ("fingerprints"):

  webmercator     everything measured in EPSG:3857
  degrees         no reprojection: lengths and areas in degrees
  crs_mismatch    no reprojection, and the task mixes a UTM GeoPackage with WGS84 layers
  fixed_utm       a UTM zone hard-coded in the script (EPSG parsed from the code)
  one_utm_large   estimate_utm_crs() on a large-extent task (one zone over tens of degrees)
  unit            right method, answer off by a power of ten (unit conversion)

An answer matches a method when it is within 1% of that method's value on every variant (counts
exactly). Answers matching none are split by the spread of value/truth across variants:
  nan_or_inf      printed nan or inf on some variant
  zero_result     printed 0 on every variant (empty filter or join)
  constant_ratio  same relative error everywhere: a location-independent mistake (misread
                  question, wrong predicate, double counting, ...)
  other           location-dependent but not one of the fingerprints

    python scripts/classify_errors.py --root data/v1 results/*.jsonl --out results/errors.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsb.audit import NonFinite, guarded  # noqa: E402
from gsb.build import read_layers  # noqa: E402
from gsb.families import BY_NAME  # noqa: E402
from gsb.harness import extract_code, matches  # noqa: E402

EPSG_UTM = re.compile(r"(?:EPSG:|epsg:|epsg=|EPSG=|crs=|to_crs\()?\b(32[67](?:0[1-9]|[1-5]\d|60))\b")
FEATURES = {
    "uses_3857": re.compile(r"3857|900913|web.?mercator", re.I),
    "estimate_utm": re.compile(r"estimate_utm_crs"),
    "geodesic": re.compile(r"\bGeod\b|geodesic|haversine|great.?circle|vincenty", re.I),
    "equal_area": re.compile(r"\b(6933|8857|54009|3410|cea|laea|aea|moll)\b|equal.?area|\+proj=(cea|laea|aea|moll)", re.I),
    "fixed_utm": EPSG_UTM,
    "to_crs": re.compile(r"to_crs|Transformer|pyproj\.transform"),
    "deg_to_m": re.compile(r"111[_,]?(?:0|1|3)\d\d|111\.\d+|6371|6378"),
}


def solve_all(task, root, proj_of):
    fam = BY_NAME[task["family"]]
    out = {}
    for v, info in task["variants"].items():
        layers = read_layers(root / info["dir"], task["files"])
        try:
            proj = proj_of(layers)
            out[v] = fam.solve(layers, guarded(proj) if proj else proj, task["params"])
        except Exception:  # noqa: BLE001  (NonFinite, GEOS errors on absurd coordinates)
            out[v] = None
    return out


def ident(g):  # no reprojection: work in whatever CRS the file was stored in
    return g


def fingerprints(task, root):
    fp = {"webmercator": solve_all(task, root, lambda L: (lambda g: g.to_crs("EPSG:3857"))),
          "degrees": solve_all(task, root, lambda L: ident)}
    if BY_NAME[task["family"]].large:
        def one_utm(L):
            crs = next(iter(L.values())).estimate_utm_crs()
            return lambda g: g.to_crs(crs)
        fp["one_utm_large"] = solve_all(task, root, one_utm)
    return fp


def match_all(vals, ref, answer):
    return all(ref.get(v) is not None and matches(x, ref[v], answer) for v, x in vals.items())


def power_of_ten(vals, ref):
    """k if vals == ref * 10**k (k != 0) within 1% on every variant."""
    ks = set()
    for v, x in vals.items():
        r = ref.get(v)
        if not r or x <= 0 or r <= 0:
            return None
        lg = math.log10(x / r)
        k = round(lg)
        if k == 0 or abs(lg - k) > math.log10(1.01):
            return None
        ks.add(k)
    return ks.pop() if len(ks) == 1 else None


def classify(row, task, fp, root, fixed_cache):
    vals = {v: r["value"] for v, r in row["variants"].items()}
    truth = {v: i["truth"] for v, i in task["variants"].items()}
    code = extract_code(row["text"]) or ""
    feats = sorted(k for k, rx in FEATURES.items() if rx.search(code))
    if not all(math.isfinite(x) for x in vals.values()):
        return "nan_or_inf", feats
    refs = dict(fp)
    for epsg in sorted(set(EPSG_UTM.findall(code))):
        key = (task["task_id"], epsg)
        if key not in fixed_cache:
            fixed_cache[key] = solve_all(task, root, lambda L, e=epsg: (lambda g: g.to_crs(f"EPSG:{e}")))
        refs[f"fixed_utm:{epsg}"] = fixed_cache[key]
    for name, ref in refs.items():
        if match_all(vals, ref, task["answer"]):
            if name == "degrees" and task["store_utm"] and len(task["files"]) > 1:
                return "crs_mismatch", feats  # a UTM GeoPackage combined with WGS84 layers as-is
            return name.split(":")[0], feats
    for name, ref in {"truth": truth, **refs}.items():
        k = power_of_ten(vals, ref)
        if k is not None:
            return ("unit" if name == "truth" else f"unit+{name.split(':')[0]}"), feats
    if all(x == 0 for x in vals.values()):
        return "zero_result", feats
    ratios = [vals[v] / truth[v] for v in vals if truth[v]]
    if ratios and min(ratios) > 0 and (max(ratios) / min(ratios) - 1) <= 0.01:
        return "constant_ratio", feats
    return "other", feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/v1")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    tasks = {t["task_id"]: t for t in map(json.loads, open(root / "tasks.jsonl")) if t["split"] == "test"}
    fps, fixed_cache, out = {}, {}, {}
    for f in a.files:
        rows = [r for r in map(json.loads, open(f)) if r["category"] in ("silent_wrong", "lucky")]
        labels = []
        for r in rows:
            t = tasks[r["task_id"]]
            if t["task_id"] not in fps:
                fps[t["task_id"]] = fingerprints(t, root)
            lab, feats = classify(r, t, fps[t["task_id"]], root, fixed_cache)
            labels.append({"task_id": r["task_id"], "sample": r["sample"], "family": r["family"],
                           "category": r["category"], "label": lab, "features": feats})
        out[Path(f).stem] = labels
        c = Counter(l["label"] for l in labels)
        print(f"\n{Path(f).stem}: {len(labels)} wrong answers")
        for lab, n in c.most_common():
            print(f"  {lab:20s}{n:6d}  {n / len(labels):.3f}")
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()

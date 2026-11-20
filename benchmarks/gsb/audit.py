"""Check what the three-location scoring can and cannot catch.

For each task, re-solve with a known method and score it against the stored truths:
  utm          estimate_utm_crs() per dataset (a correct method; should pass)
  webmercator  EPSG:3857 (the error seen in the pilot)
  utm30n       hard-coded EPSG:32630
  low_only     what single-location scoring would say about webmercator (low variant only)

    python -m benchmarks.gsb.audit data/v1
"""
from __future__ import annotations

import json
import sys

import numpy as np
from collections import defaultdict
from pathlib import Path

from .build import read_layers, reference_proj
from .families import BY_NAME
from .harness import matches

METHODS = {
    "utm": lambda g: g.to_crs(g.estimate_utm_crs()),
    "webmercator": lambda g: g.to_crs("EPSG:3857"),
    "utm30n": lambda g: g.to_crs("EPSG:32630"),
}


class NonFinite(ValueError):
    pass


def guarded(proj):
    def f(g):
        r = proj(g)
        if not np.isfinite(r.total_bounds).all():
            raise NonFinite
        return r
    return f


def classify(oks):
    return "correct" if all(oks) else ("silent_wrong" if not any(oks) else "lucky")


def audit_task(task: dict, root: Path) -> dict:
    fam = BY_NAME[task["family"]]
    out = {}
    for m, base in {"reference": None, **METHODS}.items():
        oks = []
        for v, info in task["variants"].items():
            layers = read_layers(root / info["dir"], task["files"])
            if m == "reference":
                proj = None if fam.large else reference_proj(layers)
            elif m == "utm":  # one UTM zone for all layers, chosen from the first layer
                crs = next(iter(layers.values())).estimate_utm_crs()
                proj = lambda g, crs=crs: g.to_crs(crs)
            else:
                proj = base
            if proj is not None:
                proj = guarded(proj)
            try:
                val = fam.solve(layers, proj, task["params"])
                oks.append(matches(val, info["truth"], task["answer"]))
            except NonFinite:  # e.g. EPSG:32630 far outside its zone; GEOS can crash on inf coordinates
                oks.append(False)
            if m == "webmercator" and v == "low":
                out["low_only"] = "correct" if oks[-1] else "wrong"
        out[m] = classify(oks)
    return out


def main(root: str):
    root = Path(root)
    tasks = [json.loads(l) for l in open(root / "tasks.jsonl")]
    table = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        r = audit_task(t, root)
        f = table[t["family"]]
        f["n"] += 1
        f["ref_pass"] += r["reference"] == "correct"
        f["utm_pass"] += r["utm"] == "correct"
        f["wm_caught"] += r["webmercator"] != "correct"
        f["wm_lucky"] += r["webmercator"] == "lucky"
        f["wm_low_only_pass"] += r["low_only"] == "correct"
        f["utm30n_caught"] += r["utm30n"] != "correct"
    cols = ["n", "ref_pass", "utm_pass", "wm_caught", "wm_lucky", "wm_low_only_pass", "utm30n_caught"]
    print("family".ljust(22) + "".join(c.rjust(17) for c in cols))
    for fam, f in table.items():
        print(fam.ljust(22) + "".join(str(f[c]).rjust(17) for c in cols))
    return table


if __name__ == "__main__":
    main(sys.argv[1])

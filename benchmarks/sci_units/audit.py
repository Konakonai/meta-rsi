"""Method-separability audit for SciUnits (`python -m benchmarks.sci_units.audit data/sci_v1`)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from benchmarks.gsb.harness import matches

from .build import read_workspace_si
from .families import BY_NAME


def _read_naive(d: Path) -> dict[str, pd.DataFrame]:
    """Naive solver ignoring metadata.json (assumes SI, NaN-only missingness, pre-sorted rows, raw dropna)."""
    out = {}
    for p in sorted(d.glob("*.csv")):
        out[p.stem] = pd.read_csv(p).dropna()
    return out


def audit_dataset(root: Path) -> dict:
    tasks = [json.loads(ln) for ln in open(root / "tasks.jsonl")]
    stats = Counter({
        "total": len(tasks),
        "ref_pass": 0,
        "naive_low_pass": 0,
        "naive_all_pass": 0,
        "non_control_total": 0,
        "non_control_naive_low_pass": 0,
        "non_control_naive_all_pass": 0,
    })

    for t in tasks:
        fam = BY_NAME[t["family"]]
        is_ctrl = fam.group == "control"
        if not is_ctrl:
            stats["non_control_total"] += 1

        ref_ok = True
        naive_ok_by_v: dict[str, bool] = {}
        for v, info in t["variants"].items():
            d = root / info["dir"]
            rv = float(fam.solve(read_workspace_si(d), t["params"]))
            if not matches(rv, info["truth"], t["answer"]):
                ref_ok = False
            try:
                nv = float(fam.solve(_read_naive(d), t["params"]))
                naive_ok_by_v[v] = bool(matches(nv, info["truth"], t["answer"]))
            except Exception:
                naive_ok_by_v[v] = False

        if ref_ok:
            stats["ref_pass"] += 1
        if naive_ok_by_v.get("low", False):
            stats["naive_low_pass"] += 1
            if not is_ctrl:
                stats["non_control_naive_low_pass"] += 1
        if all(naive_ok_by_v.values()):
            stats["naive_all_pass"] += 1
            if not is_ctrl:
                stats["non_control_naive_all_pass"] += 1

    return dict(stats)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sci_v1")
    res = audit_dataset(root)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

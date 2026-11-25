"""Calibrate E2 held-out mechanisms on scripts outside the E2 comparison (docs/NEXT_EXPERIMENTS.md, Amendment 2).

For each mechanism, runs saved Protocol A scripts of Stage 3 models (`runs/v9_rebuttal_gpu`, not part of the
unified re-evaluation that E2 compares) on its variants, and reports retention: the share of variants answered
correctly among tasks whose script was correct on `low`. Used only to decide which candidate mechanisms are
informative enough to enter E2, never on the unified harnesses.

    nix-shell --run "python scripts/rebuttal/calibrate_heldout_mech.py --mechanisms vector_z,multipart"
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.gsb.harness import matches, run_variant  # noqa: E402

SOURCES = ("eval_v2_var_base_26b_rep1", "eval_v2_var_sft_meta_rep1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", default=str(REPO / "data/v2_heldout_mech"))
    ap.add_argument("--evals", default=str(REPO / "runs/v9_rebuttal_gpu"))
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--mechanisms", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=25.0)
    a = ap.parse_args()
    root = Path(a.heldout)
    tasks = {t["task_id"]: t for t in map(json.loads, open(root / "tasks.jsonl"))}
    out = {}
    for src in a.sources.split(","):
        ev = json.loads((Path(a.evals) / f"{src}.json").read_text())
        low_ok = {r["task_id"] for r in ev["records"] if r["variants"].get("low", {}).get("correct")}
        code = {r["task_id"]: r.get("code") for r in map(json.loads, open(Path(a.evals) / f"{src}.code.jsonl"))}
        for mech in a.mechanisms.split(","):
            jobs = [(tid, t["variants"][mech]) for tid, t in tasks.items()
                    if mech in t["variants"] and t["variants"][mech]["scored"] and tid in low_ok and code.get(tid)]

            def run(j):
                tid, v = j
                r = run_variant(code[tid], root / v["dir"], a.timeout)
                return r["status"] == "ok" and matches(r["value"], v["truth"], tasks[tid]["answer"])

            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                ok = list(ex.map(run, jobs))
            n_var = sum(1 for t in tasks.values() if mech in t["variants"] and t["variants"][mech]["scored"])
            out[f"{src}|{mech}"] = {"n_variants": n_var, "n_low_ok": len(jobs), "retained": int(sum(ok)),
                                    "retention": round(sum(ok) / max(len(jobs), 1), 4)}
            print(f"{src:32s} {mech:16s} low-ok {len(jobs):4d}/{n_var:4d}  retention {sum(ok) / max(len(jobs), 1):.3f}", flush=True)
    (root / "calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

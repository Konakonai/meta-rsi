"""E1 analysis as pre-registered (docs/NEXT_EXPERIMENTS.md, E1 + Amendments 1-3). Fixed before any E1 data.

1. Data link (Amendment 2): for every task in T, oracle-correct share of `pool_m` (weak check + gate) minus that
   of `pool_w` (weak check), averaged within family; exact sign-flip over families. A positive confirmatory result
   is attributed to cleaner data only if this is positive with p < 0.05.
2. Confirmatory (Amendment 1): per replicate r, d_r = Protocol A accuracy of `bok_meta_r` - `bok_weak_r` on the
   450 test tasks; exact two-sided sign-flip over the 6 replicates; established only if p < 0.05 (6/6 same sign).
   Conditional on the realized K=16 pool.
3. Descriptive: pooled family-cluster bootstrap and t5 interval of d; Protocol B (100-task subset) and OSM;
   each arm against the bare model on the same server; silent and loud failures together.

    nix-shell --run "python scripts/rebuttal/e1_summary.py runs/v12_e1_bok"   # -> <dir>/e1_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from meta_rsi.eval.unified_summary import _cat, hier_bootstrap_ci, sign_flip_p, t_interval  # noqa: E402

REPLICATES = 6
EVALS = (("protA", 450), ("protB100", 100), ("osm_global", 96))


def pool_precision(scored: Path) -> dict:
    per_fam: dict[str, list[float]] = defaultdict(list)
    for row in map(json.loads, open(scored)):
        cands = row["candidates"]
        pw = [c for c in cands if c["low_ok"]]
        pm = [c for c in pw if c["meta_ok"]]
        if not pm:
            continue  # not in T
        per_fam[row["family"]].append(np.mean([c["oracle_ok"] for c in pm]) - np.mean([c["oracle_ok"] for c in pw]))
    fam_means = {f: float(np.mean(v)) for f, v in sorted(per_fam.items())}
    d = list(fam_means.values())
    p = sign_flip_p(d)
    return {"n_tasks_T": sum(len(v) for v in per_fam.values()), "per_family_mean": fam_means,
            "n_tasks_per_family": {f: len(v) for f, v in sorted(per_fam.items())},
            "mean_of_family_means": float(np.mean(d)) if d else None, "sign_flip_p": p,
            "positive": bool(d and np.mean(d) > 0 and p < 0.05)}


def load_eval(path: Path) -> dict[str, dict]:
    return {r["task_id"]: r for r in json.loads(path.read_text())["records"]}


def acc(recs: dict[str, dict], tids: list[str]) -> float:
    return float(np.mean([_cat(recs[t]["category"]) == "correct" for t in tids]))


def failure_rates(recs: dict[str, dict]) -> dict:
    c = Counter(_cat(r["category"]) for r in recs.values())
    n = max(len(recs), 1)
    return {"silent": (c["lucky"] + c["silent_wrong"]) / n, "loud": c["error"] / n, "correct": c["correct"] / n}


def summarize(d: Path, n_boot: int = 4000) -> dict:
    ev = d / "evals"
    rep: dict = {"pool_precision": pool_precision(d / "e1_scored_candidates.jsonl")}
    audit = d / "arms" / "e1_selection_audit.json"
    if audit.exists():
        rep["selection_audit"] = json.loads(audit.read_text())
    for tag, _ in EVALS:
        arms = {}
        for r in range(REPLICATES):
            for arm in ("weak", "meta"):
                p = ev / f"eval_{tag}_bok_{arm}_r{r}.json"
                if p.exists():
                    arms[(arm, r)] = load_eval(p)
        base_p = ev / f"eval_{tag}_base_26b.json"
        base = load_eval(base_p) if base_p.exists() else None
        reps = [r for r in range(REPLICATES) if ("weak", r) in arms and ("meta", r) in arms]
        if not reps:
            continue
        tids = sorted(set.intersection(*[set(arms[(a, r)]) for r in reps for a in ("weak", "meta")]))
        fam = np.array([arms[("weak", reps[0])][t]["family"] for t in tids])
        x_m = np.array([[float(_cat(arms[("meta", r)][t]["category"]) == "correct") for t in tids] for r in reps])
        x_w = np.array([[float(_cat(arms[("weak", r)][t]["category"]) == "correct") for t in tids] for r in reps])
        diffs = [float(x) for x in (x_m - x_w).mean(1)]
        row = {"n_tasks": len(tids), "replicates": reps, "per_replicate_diff": diffs, "diff": float(np.mean(diffs)),
               "sign_flip_p": sign_flip_p(diffs), "t_ci95": t_interval(diffs),
               "family_boot_ci95": hier_bootstrap_ci(x_m, x_w, fam, n_boot=n_boot),
               "acc_meta": [acc(arms[("meta", r)], tids) for r in reps],
               "acc_weak": [acc(arms[("weak", r)], tids) for r in reps],
               "failures_meta": [failure_rates(arms[("meta", r)]) for r in reps],
               "failures_weak": [failure_rates(arms[("weak", r)]) for r in reps]}
        if base is not None and set(tids) <= set(base):
            row["acc_base"] = acc(base, tids)
            row["failures_base"] = failure_rates(base)
        if tag == "protA":
            row["confirmatory"] = True
            row["established"] = bool(len(reps) == REPLICATES and row["sign_flip_p"] < 0.05)
        rep[tag] = row
    a = rep.get("protA")
    if a:
        pp = rep["pool_precision"]
        if not a["established"]:
            reading = "2: no added value established (interval or signs include 0)"
        elif a["diff"] > 0:
            reading = ("1: filter adds value given candidates (conditional on the pool)"
                       + ("; attributed to cleaner data" if pp["positive"] else "; NOT attributed to cleaner data"))
        else:
            reading = "3: filter hurts (points to its false rejections)"
        rep["reading"] = reading
    return rep


def render(rep: dict) -> str:
    pp = rep["pool_precision"]
    lines = [f"E1 data link: pool_m - pool_w oracle precision, mean of {len(pp['per_family_mean'])} family means "
             f"{100 * (pp['mean_of_family_means'] or 0):+.1f} pp, exact sign-flip p={pp['sign_flip_p']:.4f}, |T|={pp['n_tasks_T']}"]
    for tag, _ in EVALS:
        r = rep.get(tag)
        if not r:
            continue
        lo, hi = r["t_ci95"]
        blo, bhi = r["family_boot_ci95"]
        lines.append(f"{tag:10s} meta-weak {100 * r['diff']:+.1f} pp  per replicate "
                     f"[{', '.join(f'{100 * x:+.1f}' for x in r['per_replicate_diff'])}]  sign-flip p={r['sign_flip_p']:.4f}  "
                     f"t5[{100 * lo:+.1f}, {100 * hi:+.1f}]  family boot[{100 * blo:+.1f}, {100 * bhi:+.1f}]  "
                     f"acc meta {100 * np.mean(r['acc_meta']):.1f}% weak {100 * np.mean(r['acc_weak']):.1f}%"
                     + (f" base {100 * r['acc_base']:.1f}%" if "acc_base" in r else "")
                     + ("  CONFIRMATORY: " + ("ESTABLISHED" if r["established"] else "not established") if r.get("confirmatory") else ""))
    if "reading" in rep:
        lines.append("Reading " + rep["reading"])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", default=str(REPO / "runs/v12_e1_bok"))
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    rep = summarize(Path(a.dir), a.n_boot)
    (Path(a.dir) / "e1_summary.json").write_text(json.dumps(rep, indent=2))
    print(render(rep))


if __name__ == "__main__":
    main()

"""Aggregate multi-seed RSI Harness evolution runs (mean +/- std, paired deltas, lock-in stats)."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from .report import summarize_run


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def _fmt_ms(vals: list[float], pct: bool = True) -> str:
    if not vals:
        return "   -   "
    m, s = _mean_std(vals)
    if pct:
        return f"{m*100:4.1f}±{s*100:3.1f}%"
    return f"{m:4.2f}±{s:4.2f}"


def main(dirs: list[str]) -> None:
    rows = [summarize_run(Path(d)) for d in dirs if (Path(d) / "archive.json").exists()]
    if not rows:
        print("No completed runs found.")
        return

    print("=== PER-RUN DETAILS ===")
    print(
        f"{'Exp':<26} {'Gate':<11} {'Acc':>4} {'Best':>4} | "
        f"{'DevLow':>6} {'InLoop':>6} {'OraDev':>6} {'OraSn':>6} {'OraHO':>6} | "
        f"{'Gap':>6} {'Err':>6} {'Lucky':>6} {'Calls':>5}"
    )
    print("-" * 120)
    by_gate: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_gate[r["gate"]].append(r)
        gap_s = f"{r['proxy_gap']*100:+5.1f}%" if r["proxy_gap"] is not None else "  -  "
        def p(v): return "  -  " if v is None else f"{v*100:5.1f}%"
        print(
            f"{r['exp']:<26} {r['gate']:<11} {r['accepted']:>2}/{r['gens']:<2} g{r['best_gen']:02d}  | "
            f"{p(r['best_dev_low']):>6} {p(r['best_in_loop']):>6} {p(r['best_ora_dev']):>6} "
            f"{p(r['best_ora_seen']):>6} {p(r['best_ora_ho']):>6} | "
            f"{gap_s:>6} {p(r['best_err']):>6} {p(r['best_lucky']):>6} {r['best_calls']:5.2f}"
        )

    print("\n=== MULTI-SEED AGGREGATE (mean ± std) ===")
    print(
        f"{'Gate':<12} {'Seeds':>5} | {'DevLow':>11} {'OraDev':>11} {'OraSn':>11} {'OraHO':>11} | "
        f"{'ProxyGap':>11} {'Lucky':>11} {'Err':>11} {'LockIn':>6}"
    )
    print("-" * 120)
    for gate in ("naive", "preserve", "metamorphic"):
        grp = by_gate.get(gate, [])
        if not grp:
            continue
        dev_low = [r["best_dev_low"] for r in grp]
        ora_dev = [r["best_ora_dev"] for r in grp if r["best_ora_dev"] is not None]
        ora_sn = [r["best_ora_seen"] for r in grp if r["best_ora_seen"] is not None]
        ora_ho = [r["best_ora_ho"] for r in grp if r["best_ora_ho"] is not None]
        gaps = [r["proxy_gap"] for r in grp if r["proxy_gap"] is not None]
        lucky = [r["best_lucky"] for r in grp if r["best_lucky"] is not None]
        err = [r["best_err"] for r in grp]
        # Proxy Lock-In event: proxy gap >= 15 percentage points at best archive entry
        lockins = sum(1 for g in gaps if g >= 0.15)
        print(
            f"{gate:<12} {len(grp):>5} | "
            f"{_fmt_ms(dev_low):>11} {_fmt_ms(ora_dev):>11} {_fmt_ms(ora_sn):>11} {_fmt_ms(ora_ho):>11} | "
            f"{_fmt_ms(gaps):>11} {_fmt_ms(lucky):>11} {_fmt_ms(err):>11} {lockins:>2}/{len(grp):<3}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])

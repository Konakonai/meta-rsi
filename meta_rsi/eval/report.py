"""Compare multiple RSI harness evolution runs side-by-side."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize_run(exp_dir: Path) -> dict:
    cfg = json.loads((exp_dir / "config.json").read_text())
    arch = json.loads((exp_dir / "archive.json").read_text())
    hist = [json.loads(ln) for ln in open(exp_dir / "history.jsonl") if ln.strip()]
    entries = arch["entries"]
    g0 = entries[0]
    best = max(entries, key=lambda e: (e["in_loop_rate"], -e["dev_error_rate"], e["gen"]))
    return {
        "exp": exp_dir.name,
        "gate": cfg["gate_mode"],
        "model": cfg["task_model"],
        "gens": len(hist) - 1,
        "accepted": len(entries) - 1,
        "g0_dev_low": g0["dev_low_rate"],
        "g0_ora_dev": g0["oracle_dev_rate"],
        "g0_ora_seen": g0["oracle_test_seen_rate"],
        "g0_ora_ho": g0["oracle_test_heldout_rate"],
        "best_gen": best["gen"],
        "best_dev_low": best["dev_low_rate"],
        "best_in_loop": best["in_loop_rate"],
        "best_ora_dev": best["oracle_dev_rate"],
        "best_ora_seen": best["oracle_test_seen_rate"],
        "best_ora_ho": best["oracle_test_heldout_rate"],
        "best_err": best["dev_error_rate"],
        "best_lucky": best["oracle_dev_lucky"],
        "best_calls": best["avg_llm_calls"],
        "proxy_gap": (best["dev_low_rate"] - best["oracle_dev_rate"]) if best["oracle_dev_rate"] is not None else None,
    }


def _p(v: float | None) -> str:
    return "  -  " if v is None else f"{v * 100:4.1f}%"


def main(dirs: list[str]) -> None:
    rows = [summarize_run(Path(d)) for d in dirs]
    print(
        f"{'Exp':<22} {'Gate':<11} {'Acc':>4} {'Best':>4} | "
        f"{'DevLow':>6} {'InLoop':>6} {'OraDev':>6} {'OraSn':>6} {'OraHO':>6} | "
        f"{'Gap':>6} {'Err':>6} {'Lucky':>6} {'Calls':>5}"
    )
    print("-" * 116)
    for r in rows:
        gap_s = "  -  " if r["proxy_gap"] is None else f"{r['proxy_gap'] * 100:+4.1f}%"
        print(
            f"{r['exp']:<22} {r['gate']:<11} {r['accepted']:>2}/{r['gens']:<2} g{r['best_gen']:02d}  | "
            f"{_p(r['best_dev_low']):>6} {_p(r['best_in_loop']):>6} {_p(r['best_ora_dev']):>6} "
            f"{_p(r['best_ora_seen']):>6} {_p(r['best_ora_ho']):>6} | "
            f"{gap_s:>6} {_p(r['best_err']):>6} {_p(r['best_lucky']):>6} {r['best_calls']:5.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])

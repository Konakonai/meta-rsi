"""How often does the metamorphic gate reject correct solutions, and accept wrong ones?

Takes trajectory sets that only passed the weak in-loop check (correct on the `low` variant) and, for every
script, compares two verdicts on the same code:
  gate    meta_rsi.evaluator._run_metamorphic_check with the built-in probes plus the certified probes, i.e. the
          filter behind `base_meta` / `sft_meta` and the `meta_gate_only` evolution gate
  oracle  benchmarks.gsb.harness.score on every hidden variant (correct = right on all of them)
False rejection = gate fails a script the oracle calls correct; false acceptance = gate passes a script the
oracle calls wrong. The first failing probe is recorded, so rejections can be attributed (built-in vs
certified probe, inconsistent answer vs crash).

    nix-shell --run "python scripts/rebuttal/gate_precision.py"   # -> runs/v11_gate_precision/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.gsb.harness import score as oracle_score  # noqa: E402
from meta_rsi.evaluator import _run_metamorphic_check  # noqa: E402
from meta_rsi.training.collect_trajectories import load_probes_from_dir  # noqa: E402

OUT = REPO / "runs/v11_gate_precision"
# Populations that were NOT gate-filtered (weak filter = correct on `low` only).
POPULATIONS = {
    "base_weak": "bare seed.py on the base 26B (rebuttal Stage 1)",
    "same_teacher_weak": "stage2_coev teacher (rebuttal Stage 1)",
    "darwinx_weak": "darwinx_soft s0 teacher (Wave S4-2 SFT-Weak data)",
}
BUILTIN = {"shift_and_shuffle", "crs_storage_flip"}


def _code(row: dict) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", row["assistant"], re.S)
    return m.group(1) if m else None


def judge(row: dict, task: dict, data_root: Path, probes, timeout: float) -> dict:
    code = _code(row)
    rec = {"task_id": row["task_id"], "family": task["family"], "uses_3857": bool(row.get("uses_3857"))}
    sc = oracle_score(task, code, data_root, timeout=timeout)
    rec["oracle"] = sc["category"]
    rec["variants"] = {v: {k: r.get(k) for k in ("status", "value", "correct")} for v, r in sc["variants"].items()}
    low = sc["variants"].get("low", {})
    if low.get("status") != "ok" or not low.get("correct"):
        rec["gate"] = "low_not_reproduced"  # this environment does not reproduce the collection-time low pass
        return rec
    ok, diag, details = _run_metamorphic_check(
        code, data_root / task["variants"]["low"]["dir"], low["value"], task["answer"],
        timeout=timeout, custom_probes=probes,
    )
    rec["gate"] = "pass" if ok else "fail"
    if not ok:
        # details lists every probe run so far (inconsistency) or only the failing one (crash etc.): take the last
        name, d = list(details.items())[-1]
        rec["fail_probe"] = name
        rec["fail_kind"] = d.get("status") or "inconsistent"
        assert d.get("status") or d.get("consistent") is False, (row["task_id"], details)
        rec["fail_diag"] = diag[:240]
    rec["builtin_gate"] = "pass" if ok or rec.get("fail_probe") not in BUILTIN else "fail"
    return rec


def summarize(recs: list[dict]) -> dict:
    valid = [r for r in recs if r["gate"] in ("pass", "fail")]
    out = {"n": len(recs), "n_low_not_reproduced": len(recs) - len(valid)}
    for gate_key in ("gate", "builtin_gate"):
        conf = Counter((r[gate_key], r["oracle"] == "correct") for r in valid)
        n_cor = conf[("pass", True)] + conf[("fail", True)]
        n_wrong = conf[("pass", False)] + conf[("fail", False)]
        n_pass = conf[("pass", True)] + conf[("pass", False)]
        out[gate_key] = {
            "oracle_correct": n_cor, "oracle_wrong": n_wrong,
            "pass_correct": conf[("pass", True)], "fail_correct": conf[("fail", True)],
            "pass_wrong": conf[("pass", False)], "fail_wrong": conf[("fail", False)],
            "false_rejection_rate": conf[("fail", True)] / max(n_cor, 1),   # P(fail | oracle correct)
            "false_acceptance_rate": conf[("pass", False)] / max(n_wrong, 1),  # P(pass | oracle wrong)
            "precision_of_pass": conf[("pass", True)] / max(n_pass, 1),     # P(oracle correct | pass)
        }
    fr = [r for r in valid if r["gate"] == "fail" and r["oracle"] == "correct"]
    out["false_rejections_by_probe"] = dict(Counter(f"{r['fail_probe']}:{r['fail_kind']}" for r in fr).most_common())
    out["false_rejections_by_family"] = dict(Counter(r["family"] for r in fr).most_common())
    fa = [r for r in valid if r["gate"] == "pass" and r["oracle"] != "correct"]
    out["false_acceptances_by_family"] = dict(Counter(r["family"] for r in fa).most_common())
    out["oracle_categories"] = dict(Counter(r["oracle"] for r in valid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traj-dir", default=str(OUT / "trajectories"))
    ap.add_argument("--probe-dir", default=str(REPO / "runs/v8_s4_wave2_v2/certified_probes"))
    ap.add_argument("--data-root", default=str(REPO / "data/v2"))
    ap.add_argument("--populations", default=",".join(POPULATIONS))
    ap.add_argument("--limit", type=int, default=0, help="first N scripts per population (smoke runs)")
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--timeout", type=float, default=25.0)
    a = ap.parse_args()

    data_root = Path(a.data_root)
    tasks = {t["task_id"]: t for t in map(json.loads, open(data_root / "tasks.jsonl"))}
    probes = load_probes_from_dir(Path(a.probe_dir), OUT / "_probe_mods")
    assert probes and len(probes) == 3, f"expected 3 certified probes in {a.probe_dir}"
    summary = {"probes": ["shift_and_shuffle", "crs_storage_flip (gpkg/raster tasks)"] + [p for p, _ in probes]}
    for pop in a.populations.split(","):
        rows = [json.loads(ln) for ln in open(Path(a.traj_dir) / f"{pop}.jsonl")]
        rows = rows[: a.limit] if a.limit else rows
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            recs = list(ex.map(lambda r: judge(r, tasks[r["task_id"]], data_root, probes, a.timeout), rows))
        with open(OUT / f"gate_vs_oracle_{pop}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        s = summary[pop] = summarize(recs) | {"source": POPULATIONS.get(pop, pop), "elapsed_s": round(time.time() - t0, 1)}
        g, b = s["gate"], s["builtin_gate"]
        print(f"[gate] {pop}: n={s['n']} (low not reproduced {s['n_low_not_reproduced']}) oracle correct={g['oracle_correct']} "
              f"| full gate FRR={100 * g['false_rejection_rate']:.1f}% FAR={100 * g['false_acceptance_rate']:.1f}% "
              f"| built-in only FRR={100 * b['false_rejection_rate']:.1f}% FAR={100 * b['false_acceptance_rate']:.1f}% "
              f"({s['elapsed_s']}s)", flush=True)
    (OUT / "gate_precision_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=1)[:6000])


if __name__ == "__main__":
    main()

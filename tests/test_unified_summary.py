"""Statistics behind the unified 6-seed re-evaluation summary, on synthetic per-task records."""
from __future__ import annotations

import json

import pytest

from meta_rsi.eval.unified_summary import holm, sign_flip_p, summarize

N_TASKS, SEEDS = 10, range(6)


def _eval(cats: list[str]) -> dict:
    recs = [{"task_id": f"t{i}", "family": f"f{i // 2}", "category": c, "variants": {}} for i, c in enumerate(cats)]
    return {"ALL": {"n": len(recs)}, "usage": {"calls": len(recs), "errors": 0}, "records": recs}


def test_sign_flip_and_holm():
    assert sign_flip_p([0.1] * 6) == pytest.approx(2 / 64)  # floor with 6 seeds: all-same-sign patterns only
    assert sign_flip_p([0.1, -0.1]) == 1.0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_summarize_pairs_seeds_tasks_and_protocols(tmp_path):
    (tmp_path / "evals").mkdir()
    entries = []
    # method `m` solves t0-t5 on every seed, `ref` t0-t2; t6-t7 are lucky for `m` under Protocol A.
    m_a = ["correct"] * 6 + ["lucky"] * 2 + ["silent_wrong"] * 2
    m_b = ["correct"] * 6 + ["silent_wrong"] * 4  # live deployment exposes the lucky tasks
    ref = ["correct"] * 3 + ["silent_wrong"] * 7
    for method, a_cats, b_cats in (("m", m_a, m_b), ("ref", ref, ref)):
        for s in SEEDS:
            tag = f"{method}_s{s}"
            entries.append({"tag": tag, "method": method, "seed": s, "orig_correct": 0.5})
            (tmp_path / "evals" / f"{tag}_A_rep1.json").write_text(json.dumps(_eval(a_cats)))
            (tmp_path / "evals" / f"{tag}_B_rep1.json").write_text(json.dumps(_eval(b_cats)))
    flipped = ["silent_wrong"] + m_a[1:]  # second A repeat of m_s0 loses t0
    (tmp_path / "evals" / "m_s0_A_rep2.json").write_text(json.dumps(_eval(flipped)))
    bad = _eval(ref) | {"usage": {"calls": 10, "errors": 3}}  # backend errors: must be ignored
    (tmp_path / "evals" / "ref_s0_A_rep2.json").write_text(json.dumps(bad))
    (tmp_path / "manifest.json").write_text(json.dumps({"entries": entries, "missing": []}))

    rep = summarize(tmp_path, ref="ref", n_boot=300)
    a = rep["methods"]["m"]["A"]
    assert a["correct"]["per_seed"][0] == pytest.approx(0.55)  # mean over the two repeats of seed 0
    assert a["correct"]["per_seed"][1] == pytest.approx(0.6)
    assert a["shift_vs_orig_eval"] == pytest.approx((0.55 + 5 * 0.6) / 6 - 0.5)
    row = rep["vs_ref"]["A"]["rows"][0]
    assert row["diff"] == pytest.approx((0.25 + 5 * 0.3) / 6)
    assert row["sign_flip_p"] == pytest.approx(2 / 64)
    lo, hi = row["boot_ci95"]
    assert lo < row["diff"] < hi  # 5 families: ~8% of resamples miss every family with a gain, so lo can be <= 0
    assert rep["vs_ref"]["B"]["rows"][0]["diff"] == pytest.approx(0.3)
    fate = rep["a_to_b"]["m"]
    assert fate["correct"]["b_correct"] == 1.0 and fate["lucky"]["b_correct"] == 0.0
    assert fate["lucky"]["n"] == 2 * len(SEEDS)
    rv = rep["repeat_variability_A"]
    assert rv["pairs"] == 1 and rv["mean_task_flip_rate"] == pytest.approx(0.1)

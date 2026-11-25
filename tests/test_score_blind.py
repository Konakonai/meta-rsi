"""Blind-variant summary (scripts/rebuttal/score_blind.py) on synthetic scored records."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from meta_rsi.eval.eval_test import code_sidecar

_spec = importlib.util.spec_from_file_location(
    "score_blind", Path(__file__).resolve().parents[1] / "scripts" / "rebuttal" / "score_blind.py")
score_blind = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_blind)

BANDS = ["0-30", "0-30", "30-60", "30-60", "60-80", "60-80", "80-89", "80-89"]
LAT_DIST = [40.0, 30.0, 20.0, 2.0, 1.0, 10.0, 3.0, 25.0]


def test_sidecar_names():
    ev = Path("evals/arch_coev_s0_A_rep1.json")
    assert code_sidecar(ev).name == "arch_coev_s0_A_rep1.code.jsonl"
    assert score_blind.blind_path(code_sidecar(ev)).name == "arch_coev_s0_A_rep1.blind.json"


def test_summary_pairs_seeds_and_splits_by_stratum(tmp_path):
    root, out = tmp_path / "blind", tmp_path / "out"
    (out / "evals").mkdir(parents=True)
    root.mkdir()
    tasks = []
    for i in range(4):
        variants = {f"b{k}": {"dir": f"t{i}/b{k}", "band": BANDS[k], "probe_lat_dist": LAT_DIST[k], "truth": 1.0,
                              "scored": not (i == 0 and k == 7)} for k in range(8)}
        tasks.append({"task_id": f"t{i}", "family": f"f{i // 2}", "answer": "real", "large": False, "variants": variants})
    (root / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))

    entries = []
    for method in ("m", "ref"):
        for s in range(6):
            tag = f"{method}_s{s}"
            entries.append({"tag": tag, "method": method, "seed": s})
            recs = []
            for t in tasks:
                vs = {vn: {"status": "ok", "value": 1.0,
                           # `ref` only solves the two lower bands, `m` solves everything
                           "correct": method == "m" or v["band"] in ("0-30", "30-60")}
                      for vn, v in t["variants"].items() if v["scored"]}
                recs.append({"task_id": t["task_id"], "family": t["family"], "variants": vs})
            (out / "evals" / f"{tag}_A_rep1.blind.json").write_text(json.dumps({"records": recs}))
            designed = [{"task_id": t["task_id"], "variants": {
                "low": {"correct": True}, "high": {"correct": method == "m"}, "polar": {"correct": False}}} for t in tasks]
            (out / "evals" / f"{tag}_A_rep1.json").write_text(json.dumps({"records": designed}))
    (out / "manifest.json").write_text(json.dumps({"entries": entries, "missing": []}))

    rep = score_blind.summarize(out, root, ["ref"], n_boot=200)
    assert rep["methods"]["m"]["all"]["mean"] == pytest.approx(1.0)
    # task t0 has 7 scored variants (4 solved by ref), the others 8 (4 solved)
    assert rep["methods"]["ref"]["all"]["mean"] == pytest.approx((4 / 7 + 3 * 0.5) / 4)
    assert rep["methods"]["ref"]["band:60-80"]["mean"] == 0.0
    row = rep["vs_ref"]["ref"][0]
    assert row["method"] == "m" and row["diff"] == pytest.approx(1 - (4 / 7 + 1.5) / 4)
    assert row["sign_flip_p"] == pytest.approx(2 / 64)
    assert row["by_stratum"]["band:0-30"] == 0.0 and row["by_stratum"]["band:80-89"] == 1.0
    # probe-latitude distance bins: 0-5 holds b3, b4, b6 (one lower band, two upper) -> ref solves 1 of 3
    assert row["by_stratum"]["probe_lat_dist:0-5"] == pytest.approx(2 / 3)
    assert rep["methods"]["m"]["designed"]["mean"] == 0.5 and rep["methods"]["ref"]["designed"]["mean"] == 0.0
    assert row["by_stratum"]["designed"] == 0.5
    assert row["by_stratum_ci95"]["designed"] == (pytest.approx(0.5), pytest.approx(0.5))
    far_lo, far_hi = row["by_stratum_ci95"]["probe_lat_dist:15-90"]
    assert far_lo <= row["by_stratum"]["probe_lat_dist:15-90"] <= far_hi
    rendered = score_blind.render(rep)
    assert "Blind-variant accuracy" in rendered and ">=15deg:" in rendered

"""E1 pre-registered analysis (scripts/rebuttal/e1_summary.py) on synthetic data."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("e1_summary", Path(__file__).resolve().parents[1] / "scripts" / "rebuttal" / "e1_summary.py")
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)


def _write_eval(p: Path, cats: dict[str, str]) -> None:
    p.write_text(json.dumps({"records": [{"task_id": t, "family": t.split("-")[0], "category": c} for t, c in cats.items()]}))


def test_e1_summary_confirmatory_and_data_link(tmp_path):
    (tmp_path / "evals").mkdir()
    # pool: 4 families; gate keeps only oracle-correct candidates -> data link positive in every family
    rows = []
    for f in range(4):
        for i in range(3):
            rows.append({"task_id": f"f{f}-{i}", "family": f"f{f}", "candidates": [
                {"low_ok": True, "meta_ok": True, "oracle_ok": True},
                {"low_ok": True, "meta_ok": False, "oracle_ok": False}]})
    rows.append({"task_id": "f0-9", "family": "f0", "candidates": [{"low_ok": True, "meta_ok": False, "oracle_ok": False}]})
    (tmp_path / "e1_scored_candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tasks = [f"f{f}-{i}" for f in range(4) for i in range(5)]
    for r in range(6):
        # meta solves 12/20, weak 10/20 in every replicate -> +10 pp, 6/6 positive
        _write_eval(tmp_path / "evals" / f"eval_protA_bok_meta_r{r}.json", {t: "correct" if k < 12 else "error" for k, t in enumerate(tasks)})
        _write_eval(tmp_path / "evals" / f"eval_protA_bok_weak_r{r}.json", {t: "correct" if k < 10 else "lucky" for k, t in enumerate(tasks)})
    _write_eval(tmp_path / "evals" / "eval_protA_base_26b.json", {t: "correct" if k < 5 else "error" for k, t in enumerate(tasks)})
    rep = e1.summarize(tmp_path, n_boot=200)
    pp = rep["pool_precision"]
    assert pp["n_tasks_T"] == 12 and pp["mean_of_family_means"] == pytest.approx(0.5)
    assert pp["sign_flip_p"] == pytest.approx(2 / 16) and not pp["positive"]  # 4 families cannot reach 0.05
    a = rep["protA"]
    assert a["diff"] == pytest.approx(0.1) and a["sign_flip_p"] == pytest.approx(2 / 64) and a["established"]
    assert a["acc_base"] == pytest.approx(0.25)
    assert a["failures_weak"][0]["silent"] == pytest.approx(0.5) and a["failures_meta"][0]["loud"] == pytest.approx(0.4)
    assert rep["reading"].startswith("1:") and "NOT attributed" in rep["reading"]
    assert "ESTABLISHED" in e1.render(rep)

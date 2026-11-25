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
LATS = [5.0, 20.0, 40.0, 57.0, -61.0, 70.0, -85.0, 86.5]  # domains: equatorial, 5 x utm_valid, 2 x polar


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
        variants = {f"b{k}": {"dir": f"t{i}/b{k}", "band": BANDS[k], "probe_lat_dist": LAT_DIST[k], "lat": LATS[k], "truth": 1.0,
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
    # Addendum 2 strata: mechanism domains and |lat| distance; `ref` solves only |lat| < 60
    assert row["by_stratum"]["domain:equatorial"] == 0.0 and row["by_stratum"]["domain:polar"] == 1.0
    # |lat| distance 0-5: b3 (57), b4 (-61), b6 (-85), b7 (86.5); ref solves b3 only (t0 has no b7)
    assert row["by_stratum"]["abs_lat_dist:0-5"] == pytest.approx(1 - (1 / 3 + 3 * 0.25) / 4)
    assert "all_areaw" in rep["methods"]["ref"] and "t_ci_bonf" in row
    # area weights: a band's share of the sphere split over the task's scored variants in it; ref solves 0-30, 30-60
    a = score_blind.BAND_AREA
    full = (a["0-30"] + a["30-60"]) / sum(a.values())
    assert rep["methods"]["ref"]["all_areaw"]["mean"] == pytest.approx(full)  # t0's unscored 80-89 variant: same share
    assert rep["methods"]["m"]["all_areaw"]["mean"] == pytest.approx(1.0)
    assert "domain:polar" in row["by_stratum_t95"]


def test_heldout_summary_strata_and_confirmatory(tmp_path):
    """E2 held-out mechanisms: strata by mechanism / transfer, matched tasks, low-conditioned, and the confirmatory."""
    root, out = tmp_path / "v2_heldout_mech", tmp_path / "out"
    (out / "evals").mkdir(parents=True)
    root.mkdir()
    layout = {"t0": ["us_ft", "albers", "nodata_nan"], "t1": ["us_ft", "albers", "nodata_nan"],
              "t2": ["us_ft"], "t3": ["csv_lon360"]}
    transfer = {"us_ft": "near", "albers": "near", "nodata_nan": "far", "csv_lon360": "far"}  # labels as in the data
    tasks = [{"task_id": tid, "family": f"f{i // 2}", "answer": "real", "large": False,
              "variants": {m: {"dir": f"{tid}/{m}", "lat": 10.0, "lon": -20.0, "mechanism": m, "transfer": transfer[m],
                               "truth": 1.0, "scored": True} for m in ms}}
             for i, (tid, ms) in enumerate(layout.items())]
    (root / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    assert score_blind.is_heldout({t["task_id"]: t for t in tasks})
    assert score_blind.blind_path(Path("e/naive_s0_A_rep1.code.jsonl"), ".heldout.json").name == "naive_s0_A_rep1.heldout.json"

    entries = []
    for method in ("arch_coev", "naive"):
        for s in range(6):
            tag = f"{method}_s{s}"
            entries.append({"tag": tag, "method": method, "seed": s})
            # `naive` survives only the near-transfer mechanisms; `arch_coev` survives everything
            recs = [{"task_id": t["task_id"], "family": t["family"], "variants": {
                m: {"status": "ok", "value": 1.0, "correct": method == "arch_coev" or v["transfer"] == "near"}
                for m, v in t["variants"].items()}} for t in tasks]
            (out / "evals" / f"{tag}_A_rep1.heldout.json").write_text(json.dumps({"records": recs}))
            low = [{"task_id": t["task_id"], "variants": {"low": {"correct": method == "arch_coev" or t["task_id"] == "t0"}}}
                   for t in tasks]
            (out / "evals" / f"{tag}_A_rep1.json").write_text(json.dumps({"records": low}))
            # a blind file of the same eval must not leak into the held-out summary
            (out / "evals" / f"{tag}_A_rep1.blind.json").write_text(json.dumps({"records": []}))
    (out / "manifest.json").write_text(json.dumps({"entries": entries, "missing": []}))

    rep = score_blind.summarize(out, root, ["naive"], n_boot=200)
    assert rep["kind"] == "heldout"
    nv = rep["methods"]["naive"]
    assert nv["all"]["mean"] == pytest.approx((2 / 3 + 2 / 3 + 1 + 0) / 4)
    assert nv["transfer:near"]["mean"] == pytest.approx(1.0)
    assert nv["transfer:far"]["mean"] == pytest.approx(0.0)
    assert nv["mech:csv_lon360"]["mean"] == pytest.approx(0.0)
    assert nv["matched:near"]["mean"] == pytest.approx(1.0) and nv["matched:far"]["mean"] == pytest.approx(0.0)
    assert "band:0-30" not in nv and "all_areaw" not in nv
    # lowok: naive got only t0 right on `low`
    assert nv["lowok:transfer:far"]["mean"] == pytest.approx(0.0)
    assert rep["methods"]["arch_coev"]["lowok:transfer:far"]["mean"] == pytest.approx(1.0)
    assert nv["low"]["mean"] == pytest.approx(1 / 4)  # naive solves only t0 on `low`
    assert not any(k.startswith("drop:") for k in nv)

    row = rep["vs_ref"]["naive"][0]
    assert row["by_stratum"]["transfer:far"] == pytest.approx(1.0)
    assert row["by_stratum"]["transfer:near"] == pytest.approx(0.0)
    c = rep["confirmatory"]
    assert (c["method"], c["ref"], c["stratum"]) == ("arch_coev", "naive", "transfer:far")
    assert c["diff"] == pytest.approx(1.0) and c["sign_flip_p"] == pytest.approx(2 / 64) and c["established"]
    # bothlow: only t0 is solved on `low` by both; its far variant (nodata_nan, labelled far here): arch 1, naive 0
    assert row["bothlow"]["transfer:far"]["diff"] == pytest.approx(1.0)
    assert row["bothlow"]["transfer:near"]["diff"] == pytest.approx(0.0)
    assert not any(k.startswith("lowok:") for k in row["by_stratum"])
    assert c["bothlow_diff"] == pytest.approx(1.0) and c["lowok"] == [pytest.approx(1.0), pytest.approx(0.0)]
    assert not c["uninformative"] and c["robustness_transfer"]
    text = score_blind.render(rep)
    assert "CONFIRMATORY (E2)" in text and "ESTABLISHED" in text and "csv_lon360" in text


def test_score_is_incremental_when_a_mechanism_is_added(tmp_path, monkeypatch):
    """Adding a mechanism to the variant set scores only the new (task, variant) pairs and keeps the old results."""
    import argparse
    root, out = tmp_path / "v2_heldout_mech", tmp_path / "out"
    (out / "evals").mkdir(parents=True)
    root.mkdir()
    v = lambda m, tr: {"dir": f"t0/{m}", "lat": 1.0, "lon": 2.0, "mechanism": m, "transfer": tr, "truth": 1.0, "scored": True}
    task = {"task_id": "t0", "family": "f", "answer": "real", "large": False,
            "variants": {"us_ft": v("us_ft", "near"), "raster_southup": v("raster_southup", "far")}}
    (root / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    (out / "manifest.json").write_text(json.dumps({"entries": [{"tag": "naive_s0", "method": "naive", "seed": 0}]}))
    (out / "evals" / "naive_s0_A_rep1.code.jsonl").write_text(
        json.dumps({"task_id": "t0", "code": "print(1)"}) + "\n" + json.dumps({"task_id": "t9", "code": None}) + "\n")
    old = {"status": "ok", "value": 2.0, "correct": False}
    (out / "evals" / "naive_s0_A_rep1.heldout.json").write_text(json.dumps(
        {"records": [{"task_id": "t0", "family": "f", "variants": {"us_ft": old}}]}))
    calls = []
    monkeypatch.setattr(score_blind, "run_variant", lambda code, d, timeout: calls.append(d.name) or {"status": "ok", "value": 1.0})
    score_blind.cmd_score(argparse.Namespace(out=str(out), blind=str(root), workers=2, timeout=5.0, suffix="", prioritize=""))
    assert calls == ["raster_southup"]
    rec = json.loads((out / "evals" / "naive_s0_A_rep1.heldout.json").read_text())["records"][0]["variants"]
    assert rec["us_ft"] == old and rec["raster_southup"]["correct"] is True
    calls.clear()
    score_blind.cmd_score(argparse.Namespace(out=str(out), blind=str(root), workers=2, timeout=5.0, suffix="", prioritize=""))
    assert calls == []

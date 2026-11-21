"""Unit and end-to-end tests for meta_rsi (metamorphic invariance, gating, sandbox, mock evolution)."""
from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from benchmarks.gsb.build import read_layers, reference_proj
from benchmarks.gsb.families import BY_NAME
from benchmarks.gsb.harness import matches
from meta_rsi.archive import ArchiveEntry, evaluate_gate
from meta_rsi.evaluator import _run_metamorphic_check, load_task_splits
from meta_rsi.evolve import run_evolution
from meta_rsi.infra.colab_g4 import pack_bundle
from meta_rsi.llm import LLMClient
from meta_rsi.metamorphic import perturb_workspace
from meta_rsi.sandbox import run_python_sandbox, validate_harness_source

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "v1"


def test_metamorphic_preserves_ground_truth_across_all_21_families():
    """Label-free perturbation T(low) MUST keep ground-truth answers invariant on all 21 families."""
    tasks = [json.loads(line) for line in open(DATA_ROOT / "tasks.jsonl")]
    seen = {}
    for t in tasks:
        if t["family"] not in seen:
            seen[t["family"]] = t
    assert len(seen) == 21

    for fam_name, task in sorted(seen.items()):
        fam = BY_NAME[fam_name]
        low_info = task["variants"]["low"]
        low_dir = DATA_ROOT / low_info["dir"]
        with tempfile.TemporaryDirectory() as tmp:
            pert_dir = Path(tmp) / "pert"
            perturb_workspace(low_dir, pert_dir, target_lat=58.3, target_lon=6.0, shuffle_rows=True, seed=7)
            layers = read_layers(pert_dir, task["files"])
            proj = None if fam.large else reference_proj(layers)
            val = fam.solve(layers, proj, task["params"])
        assert matches(val, low_info["truth"], task["answer"]), (
            f"Invariance broken on {fam_name}: orig={low_info['truth']}, pert={val}"
        )


def test_metamorphic_catches_webmercator_without_ground_truth():
    """On a local task near the equator, EPSG:3857 passes `low` truth, yet breaks invariance on T(low)."""
    tasks = [json.loads(line) for line in open(DATA_ROOT / "tasks.jsonl")]
    t = next(
        x
        for x in tasks
        if x["family"] == "area_by_class"
        and x["split"] == "train"
        and abs(x["variants"]["low"]["lat"]) < 2.5
    )
    fam = BY_NAME[t["family"]]
    low_dir = DATA_ROOT / t["variants"]["low"]["dir"]
    orig_layers = read_layers(low_dir, t["files"])
    wm_orig = fam.solve(orig_layers, lambda g: g.to_crs("EPSG:3857"), t["params"])
    # Near equator, EPSG:3857 passes the weak `low` check!
    assert matches(wm_orig, t["variants"]["low"]["truth"], t["answer"])

    # Under label-free T(low), EPSG:3857 diverges wildly (>3x) -> caught with ZERO labels!
    with tempfile.TemporaryDirectory() as tmp:
        pert_dir = Path(tmp) / "pert"
        perturb_workspace(low_dir, pert_dir, target_lat=58.3, target_lon=6.0)
        pert_layers = read_layers(pert_dir, t["files"])
        wm_pert = fam.solve(pert_layers, lambda g: g.to_crs("EPSG:3857"), t["params"])
    assert not matches(wm_pert, wm_orig, t["answer"])
    assert wm_pert / wm_orig > 2.5


def test_validate_harness_source_security_and_contract():
    good = "from pathlib import Path\ndef solve(task_prompt, work_dir, llm_call, run_python):\n    return {'code': None, 'trace': [], 'turns': 0}\n"
    assert validate_harness_source(good)[0]

    bad_net = "import socket\ndef solve(a, b, c, d): return {}\n"
    assert not validate_harness_source(bad_net)[0]

    bad_eval = "def solve(a, b, c, d):\n    eval('1')\n    return {}\n"
    assert not validate_harness_source(bad_eval)[0]

    bad_sig = "def solve(a, b): return {}\n"
    assert not validate_harness_source(bad_sig)[0]


def test_gate_modes_reject_regression_and_metamorphic_hacks():
    parent = ArchiveEntry(
        gen=0,
        parent_gen=None,
        harness_path="gen_000/harness.py",
        rationale="seed",
        dev_low_rate=0.5,
        in_loop_rate=0.5,
        dev_error_rate=0.2,
        avg_llm_calls=1.0,
        solved_ids=["t1"],
    )
    # Candidate improves dev_low average (0.5 -> 0.75) by solving t2, t3 via lucky hack, while regressing t1!
    cand_eval = {
        "summary": {
            "n_tasks": 4,
            "dev_low_rate": 0.75,
            "meta_pass_rate": 0.25,
            "in_loop_rate": 0.75,
            "dev_error_rate": 0.25,
            "avg_llm_calls": 1.0,
            "solved_ids_dev_low": ["t2", "t3", "t4"],
            "solved_ids_in_loop": ["t2", "t3", "t4"],
        },
        "records": [
            {"task_id": "t1", "dev_low_ok": False, "meta_checked": False, "meta_consistent": False},
            {"task_id": "t2", "dev_low_ok": True, "meta_checked": True, "meta_consistent": False},
            {"task_id": "t3", "dev_low_ok": True, "meta_checked": True, "meta_consistent": False},
            {"task_id": "t4", "dev_low_ok": True, "meta_checked": True, "meta_consistent": True},
        ],
    }
    # Naive accepts because scalar average went up (0.5 -> 0.75).
    assert evaluate_gate("naive", parent, cand_eval).accepted
    # Preserve-and-extend rejects because t1 regressed (when stepping-stone disabled)!
    dec_p = evaluate_gate("preserve", parent, cand_eval, max_regress=0, allow_stepping_stone=False)
    assert not dec_p.accepted
    assert dec_p.regressed == ["t1"]
    # With stepping-stone enabled, +3 solved / -1 regressed enters archive as an alternative lineage.
    dec_ss = evaluate_gate("preserve", parent, cand_eval, max_regress=0, allow_stepping_stone=True)
    assert dec_ss.accepted
    assert "stepping-stone" in dec_ss.reason

    # Now suppose t1 stays solved, and t2 passes dev_low ONLY by lucky hack (meta_consistent=False).
    cand_lucky = {
        "summary": {
            "n_tasks": 2,
            "dev_low_rate": 1.0,
            "meta_pass_rate": 0.5,
            "in_loop_rate": 0.5,  # metamorphic mode only counts dev_low AND meta_consistent
            "dev_error_rate": 0.0,
            "avg_llm_calls": 1.0,
            "solved_ids_dev_low": ["t1", "t2"],
            "solved_ids_in_loop": ["t1"],
        },
        "records": [
            {"task_id": "t1", "dev_low_ok": True, "meta_checked": True, "meta_consistent": True},
            {"task_id": "t2", "dev_low_ok": True, "meta_checked": True, "meta_consistent": False},
        ],
    }
    dec_m = evaluate_gate("metamorphic", parent, cand_lucky, max_regress=0)
    assert not dec_m.accepted
    assert dec_m.meta_blocked == ["t2"]


def test_end_to_end_mock_evolution():
    """Verify full 1-generation evolution loop produces archive, diff, history, and oracle metrics."""
    # Task mock: emits EPSG:3857 when prompt has no hint, emits estimate_utm_crs when harness adds `USE_UTM`.
    def mock_task_llm(system: str, user: str) -> str:
        use_utm = "USE_UTM" in (system + user)
        crs_line = "g = g.to_crs(g.estimate_utm_crs())" if use_utm else "g = g.to_crs('EPSG:3857')"
        return (
            "```python\n"
            "import glob, geopandas as gpd\n"
            "f = sorted(glob.glob('*.geojson') + glob.glob('*.gpkg'))[0]\n"
            "g = gpd.read_file(f)\n"
            f"{crs_line}\n"
            "print(g.geometry.area.sum())\n"
            "```"
        )

    mutated_harness = '''
from pathlib import Path
import re
def solve(task_prompt, work_dir, llm_call, run_python):
    reply = llm_call("USE_UTM", task_prompt)
    blocks = re.findall(r"```(?:python)?\\s*\\n(.*?)```", reply, re.S)
    return {"code": blocks[-1].strip() if blocks else None, "trace": [], "turns": 1}
'''

    def mock_meta_llm(system: str, user: str) -> str:
        return f"HYPOTHESIS: Switch to local UTM projection.\n```python\n{mutated_harness}\n```"

    task_llm = LLMClient("mock", "mock-task", mock_fn=mock_task_llm)
    meta_llm = LLMClient("mock", "mock-meta", mock_fn=mock_meta_llm)

    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp_test"
        arch = run_evolution(
            exp_dir=exp_dir,
            data_root=DATA_ROOT,
            gate_mode="metamorphic",
            generations=1,
            task_llm=task_llm,
            meta_llm=meta_llm,
            dev_per_family=1,
            test_seen_per_family=0,
            test_heldout_per_family=0,
            limit_dev=2,
            workers=1,
        )
        assert (exp_dir / "archive.json").exists()
        assert (exp_dir / "history.jsonl").exists()
        assert (exp_dir / "gen_001" / "diff.patch").exists()
        assert len(arch.entries) >= 1


def test_seed_harness_and_subpackages_layout():
    """Verify packaged seed harness passes AST check and Colab bundler includes both packages."""
    repo_root = Path(__file__).resolve().parents[1]
    seed_path = repo_root / "meta_rsi" / "harnesses" / "seed.py"
    ok, reason = validate_harness_source(seed_path.read_text())
    assert ok, reason

    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "bundle.tgz"
        n = pack_bundle(repo_root, DATA_ROOT, tgz, dev_per_family=1, test_seen_per_family=0, test_heldout_per_family=0)
        assert n == 18
        with tarfile.open(tgz, "r:gz") as tf:
            names = set(tf.getnames())
        assert "meta_rsi/evolve.py" in names
        assert "meta_rsi/harnesses/seed.py" in names
        assert "benchmarks/gsb/harness.py" in names


def test_polar_ood_preserves_truth_and_breaks_utm():
    """Polar OOD relocations (>84N / <80S) keep AEQD ground truth invariant while breaking estimate_utm_crs()."""
    tasks = [json.loads(line) for line in open(DATA_ROOT / "tasks.jsonl")]
    seen = {}
    for t in tasks:
        if not BY_NAME[t["family"]].large and t["family"] not in seen:
            seen[t["family"]] = t
    assert len(seen) == 19

    utm_exceptions = 0
    for fam_name, task in sorted(seen.items()):
        fam = BY_NAME[fam_name]
        low_info = task["variants"]["low"]
        low_dir = DATA_ROOT / low_info["dir"]
        for lat, lon, flip in [(86.5, 25.0, False), (-85.5, -40.0, True)]:
            with tempfile.TemporaryDirectory() as tmp:
                pert_dir = Path(tmp) / "polar"
                perturb_workspace(low_dir, pert_dir, target_lat=lat, target_lon=lon, shuffle_rows=True, flip_gpkg_crs=flip, seed=11)
                layers = read_layers(pert_dir, task["files"])
                val = fam.solve(layers, reference_proj(layers), task["params"])
                assert matches(val, low_info["truth"], task["answer"]), (
                    f"Polar invariance broken on {fam_name} @ ({lat},{lon}): orig={low_info['truth']}, pert={val}"
                )
                try:
                    next(iter(layers.values())).to_crs("EPSG:4326").estimate_utm_crs()
                except RuntimeError:
                    utm_exceptions += 1
    assert utm_exceptions == 19 * 2


def test_metamorphic_diag_is_domain_neutral_by_default():
    """Default metamorphic diagnostics must not leak GIS/CRS hints unless explicitly opted in."""
    tasks = [json.loads(line) for line in open(DATA_ROOT / "tasks.jsonl")]
    t = next(
        x
        for x in tasks
        if x["family"] == "area_by_class"
        and x["split"] == "train"
        and abs(x["variants"]["low"]["lat"]) < 2.5
    )
    low_dir = DATA_ROOT / t["variants"]["low"]["dir"]
    wm_script = (
        "import glob, geopandas as gpd\n"
        "f = sorted(glob.glob('*.geojson') + glob.glob('*.gpkg'))[0]\n"
        "g = gpd.read_file(f).to_crs('EPSG:3857')\n"
        "print(g.geometry.area.sum())\n"
    )
    orig = float(run_python_sandbox(wm_script, low_dir)["stdout"].strip().splitlines()[-1])

    ok_neutral, diag_neutral, _ = _run_metamorphic_check(
        wm_script, low_dir, orig, t["answer"], include_domain_hint=False
    )
    assert not ok_neutral
    assert "EPSG:3857" not in diag_neutral
    assert "UTM" not in diag_neutral

    ok_hint, diag_hint, _ = _run_metamorphic_check(
        wm_script, low_dir, orig, t["answer"], include_domain_hint=True
    )
    assert not ok_hint
    assert "EPSG:3857" in diag_hint


def test_oracle_gate_and_caught_exception_tracking():
    """Oracle gate accepts new oracle-solved tasks; sandbox tracks swallowed exceptions when enabled."""
    parent = ArchiveEntry(
        gen=0,
        parent_gen=None,
        harness_path="gen_000/harness.py",
        rationale="seed",
        dev_low_rate=0.5,
        in_loop_rate=0.25,
        dev_error_rate=0.0,
        avg_llm_calls=1.0,
        solved_ids=["t1"],
    )
    cand_eval = {
        "summary": {
            "n_tasks": 4,
            "dev_low_rate": 0.5,
            "meta_pass_rate": None,
            "in_loop_rate": 0.5,
            "dev_error_rate": 0.0,
            "avg_llm_calls": 1.0,
            "solved_ids_dev_low": ["t1", "t2"],
            "solved_ids_in_loop": ["t1", "t2"],
        },
        "records": [
            {"task_id": "t1", "dev_low_ok": True, "meta_checked": False, "meta_consistent": False},
            {"task_id": "t2", "dev_low_ok": True, "meta_checked": False, "meta_consistent": False},
        ],
    }
    dec = evaluate_gate("oracle", parent, cand_eval, max_regress=0)
    assert dec.accepted
    assert "oracle-preserve-and-extend" in dec.reason

    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp) / "w"
        w.mkdir()
        swallow_code = "try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    x = 0\nprint(x)\n"
        r = run_python_sandbox(swallow_code, w, track_caught=True)
        assert r["returncode"] == 0
        assert any("ZeroDivisionError" in e for e in r["caught_exceptions"])


def test_strict_anonymization_minimal_prompt_and_new_harnesses():
    """Verify zero residual domain leakage in anonymized diagnostics, minimal prompt, and baseline harnesses."""
    from meta_rsi.meta_agent import MINIMAL_META_SYSTEM_PROMPT, build_failure_digest

    repo_root = Path(__file__).resolve().parents[1]
    for name in ("handcrafted.py", "runtime_meta.py"):
        ok, reason = validate_harness_source((repo_root / "meta_rsi" / "harnesses" / name).read_text())
        assert ok, f"{name} failed AST check: {reason}"

    for forbidden in ("geopandas", "pyproj", "shapely", "workspace inspection", "skill contamination"):
        assert forbidden not in MINIMAL_META_SYSTEM_PROMPT.lower()

    tasks = [json.loads(line) for line in open(DATA_ROOT / "tasks.jsonl")]
    t = next(
        x for x in tasks
        if x["family"] == "area_by_class" and x["split"] == "train" and abs(x["variants"]["low"]["lat"]) < 2.5
    )
    low_dir = DATA_ROOT / t["variants"]["low"]["dir"]
    wm_script = (
        "import glob, geopandas as gpd\n"
        "f = sorted(glob.glob('*.geojson') + glob.glob('*.gpkg'))[0]\n"
        "g = gpd.read_file(f).to_crs('EPSG:3857')\n"
        "print(g.geometry.area.sum())\n"
    )
    orig = float(run_python_sandbox(wm_script, low_dir)["stdout"].strip().splitlines()[-1])
    ok_neutral, diag_neutral, _ = _run_metamorphic_check(
        wm_script, low_dir, orig, t["answer"], include_domain_hint=False
    )
    assert not ok_neutral
    assert "probe_1" in diag_neutral
    for leak in ("shift_and_shuffle", "crs_storage_flip", "local_aeqd_shift", "aeqd", "EPSG:3857", "UTM"):
        assert leak.lower() not in diag_neutral.lower(), f"Leaked token {leak!r} in {diag_neutral}"

    # Verify meta_gate_only hides metamorphic diagnostics completely
    fake_eval = {
        "summary": {
            "n_tasks": 2,
            "dev_low_rate": 0.5,
            "meta_pass_rate": 0.0,
            "in_loop_rate": 0.0,
            "dev_error_rate": 0.5,
            "avg_llm_calls": 1.0,
        },
        "records": [
            {
                "task_id": "t1",
                "family": "area_by_class",
                "dev_low_status": "ok",
                "dev_low_ok": True,
                "dev_low_value": 100.0,
                "dev_low_truth": 100.0,
                "meta_checked": True,
                "meta_consistent": False,
                "meta_diag": "METAMORPHIC_INCONSISTENCY (probe_1): ratio=3.5x",
                "code": "print(100)",
            },
            {
                "task_id": "t2",
                "family": "large_route_length",
                "dev_low_status": "exception",
                "dev_low_ok": False,
                "dev_low_stderr": "KeyError: 'id'",
                "meta_checked": False,
                "meta_consistent": False,
                "meta_diag": "",
                "code": "print(x)",
            },
        ],
    }
    dig_only = build_failure_digest(fake_eval, "meta_gate_only", minimal_prompt=True, hide_meta_diag=True)
    assert "METAMORPHIC_INCONSISTENCY" not in dig_only
    assert "ratio=3.5x" not in dig_only
    assert "family=" not in dig_only


def test_verifier_conservation_auditor_accepts_polar_and_rejects_bad():
    """Zero-label conservation auditor accepts valid polar AEQD probe and rejects vacuous/distorted probes."""
    from meta_rsi.verifier_agent import certify_perturbation_operator

    tasks = [json.loads(line) for line in open(DATA_ROOT / "tasks.jsonl")]
    t = next(x for x in tasks if x["family"] == "area_by_class" and x["split"] == "train")
    low_dir = DATA_ROOT / t["variants"]["low"]["dir"]
    utm_script = (
        "import glob, geopandas as gpd\n"
        "f = sorted(glob.glob('*.geojson') + glob.glob('*.gpkg'))[0]\n"
        "g = gpd.read_file(f)\n"
        "g = g.to_crs(g.estimate_utm_crs())\n"
        "print(g.geometry.area.sum())\n"
    )
    orig = float(run_python_sandbox(utm_script, low_dir)["stdout"].strip().splitlines()[-1])
    passing_cases = [(low_dir, utm_script, orig, t["answer"])]

    # 1. Valid polar AEQD operator -> MUST be certified!
    good_polar = '''
import shutil
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer

def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in (".geojson", ".gpkg"):
            g = gpd.read_file(p)
            g4326 = g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")
            layers.append((p, ext, g4326))
        elif ext == ".csv":
            df = pd.read_csv(p)
            if {"lon", "lat"}.issubset(df.columns):
                g = gpd.GeoDataFrame(df.drop(columns=["lon", "lat"]), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
                layers.append((p, ".csv", g))
            else:
                shutil.copy2(p, dst_dir / p.name)
        else:
            shutil.copy2(p, dst_dir / p.name)
    b = np.array([g.total_bounds for _, _, g in layers])
    cx, cy = 0.5 * float(b[:,0].min() + b[:,2].max()), 0.5 * float(b[:,1].min() + b[:,3].max())
    c1 = CRS.from_proj4(f"+proj=aeqd +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m +no_defs")
    c2 = CRS.from_proj4("+proj=aeqd +lat_0=86.2 +lon_0=15.0 +datum=WGS84 +units=m +no_defs")
    t1 = Transformer.from_crs("EPSG:4326", c1, always_xy=True)
    t2 = Transformer.from_crs(c2, "EPSG:4326", always_xy=True)
    def xf(geoms, tr):
        return [shapely.transform(g, lambda c: np.column_stack(tr.transform(c[:,0], c[:,1]))) for g in geoms]
    for p, ext, g4326 in layers:
        gn = gpd.GeoDataFrame(g4326.drop(columns="geometry").copy(), geometry=xf(xf(list(g4326.geometry), t1), t2), crs="EPSG:4326")
        if ext == ".csv":
            df = pd.DataFrame(gn.drop(columns="geometry"))
            df["lon"], df["lat"] = gn.geometry.x.round(7), gn.geometry.y.round(7)
            df.to_csv(dst_dir / p.name, index=False)
        elif ext == ".gpkg":
            gn.to_file(dst_dir / p.name, driver="GPKG")
        else:
            gn.to_file(dst_dir / p.name, driver="GeoJSON")
    return {"mode": "polar_north"}
'''
    res_good = certify_perturbation_operator(good_polar, [low_dir], passing_cases)
    assert res_good["certified"], f"Expected valid polar probe certified, got: {res_good}"
    assert res_good["metrics"]["caught"] == 1

    # 2. Identity vacuous operator -> MUST be rejected at non_vacuity!
    vacuous = '''
import shutil
from pathlib import Path
def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)
    return {"mode": "noop"}
'''
    res_vac = certify_perturbation_operator(vacuous, [low_dir], passing_cases)
    assert not res_vac["certified"] and res_vac["stage"] == "non_vacuity"

    # 3. Non-isometric distorted operator (scales geometry 2x) -> MUST be rejected at conservation!
    distorted = '''
import shutil
from pathlib import Path
import geopandas as gpd
import shapely.affinity
def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() in (".geojson", ".gpkg"):
            g = gpd.read_file(p)
            g4326 = g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")
            g4326["geometry"] = [shapely.affinity.scale(x, xfact=2.0, yfact=2.0) for x in g4326.geometry]
            g4326.to_file(dst_dir / p.name, driver="GPKG" if p.suffix.lower() == ".gpkg" else "GeoJSON")
        elif p.is_file():
            shutil.copy2(p, dst_dir / p.name)
    return {"mode": "scale_2x"}
'''
    res_bad = certify_perturbation_operator(distorted, [low_dir], passing_cases)
    assert not res_bad["certified"] and res_bad["stage"] == "conservation"


def test_extract_hypothesis_preserves_channel_tag_in_code():
    """Stripping reasoning tags must never split on `<channel|>` inside harness source literals."""
    from meta_rsi.meta_agent import _extract_hypothesis_and_code

    raw = (
        "thought\nI should improve the harness.\n<channel|>"
        "HYPOTHESIS: Keep channel stripping intact.\n"
        "```python\n"
        "def extract_code(text):\n"
        "    for tag in ('</think>', '<channel|>'):\n"
        "        if tag in text:\n"
        "            text = text.rsplit(tag, 1)[1]\n"
        "    return text\n"
        "\n"
        "def solve(task_prompt, workdir, llm, exec_fn):\n"
        "    return None, {'llm_calls': 0}\n"
        "```\n"
    )
    hyp, code = _extract_hypothesis_and_code(raw)
    assert "Keep channel stripping intact" in hyp
    assert code is not None
    assert "def extract_code(text):" in code
    assert "'<channel|>'" in code
    assert "def solve(" in code


def test_extract_handles_multiple_fenced_blocks():
    """When a response includes an example snippet before the target function block, extract the valid target block."""
    from meta_rsi.meta_agent import _extract_hypothesis_and_code
    from meta_rsi.verifier_agent import _extract_hyp_and_code

    raw_v = (
        "thought\nLet's inspect the failing line:\n```python\ng = g.to_crs('EPSG:3857')\n```\n"
        "<channel|>"
        "HYPOTHESIS: Test an invariant coordinate transform.\n"
        "Example snippet from solver:\n"
        "```python\nx = 1 + 1\n```\n"
        "Full operator:\n"
        "```python\nfrom pathlib import Path\n\ndef perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:\n    return {'mode': 'test'}\n```\n"
    )
    hyp_v, code_v = _extract_hyp_and_code(raw_v)
    assert "invariant coordinate transform" in hyp_v
    assert code_v is not None and "def perturb(" in code_v and "x = 1 + 1" not in code_v

    raw_m = (
        "HYPOTHESIS: Add repair loop.\n"
        "Note this snippet:\n```python\ny = 2\n```\n"
        "```python\ndef solve(task_prompt, work_dir, llm_call, run_python):\n    return {'code': None, 'trace': [], 'turns': 0}\n```\n"
    )
    hyp_m, code_m = _extract_hypothesis_and_code(raw_m)
    assert "Add repair loop" in hyp_m
    assert code_m is not None and "def solve(" in code_m and "y = 2" not in code_m


def test_llm_judge_gate_blocks_shortcut_and_formats_digest():
    """Verify --gate llm_judge rejects judge-flagged scripts and feeds rubric reason into failure digest."""
    from meta_rsi.evaluator import _run_llm_judge_check
    from meta_rsi.meta_agent import build_failure_digest

    def mock_judge(system: str, user: str) -> str:
        if "EPSG:3857" in user:
            return '{"verdict": "FAIL", "reason": "Uses Web Mercator EPSG:3857 for area calculation."}'
        return '{"verdict": "PASS", "reason": "Uses dynamic local projection."}'

    j_llm = LLMClient("mock", "mock-judge", mock_fn=mock_judge)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "parcels.geojson").write_text("{}")
        ok_bad, diag_bad = _run_llm_judge_check(j_llm, "Compute area", d, "g = g.to_crs('EPSG:3857')")
        ok_good, diag_good = _run_llm_judge_check(j_llm, "Compute area", d, "g = g.to_crs(aeqd)")
        assert not ok_bad and "EPSG:3857" in diag_bad
        assert ok_good

    parent = ArchiveEntry(
        gen=0,
        parent_gen=None,
        harness_path="gen_000/harness.py",
        rationale="seed",
        dev_low_rate=0.5,
        in_loop_rate=0.0,
        dev_error_rate=0.0,
        avg_llm_calls=1.0,
        solved_ids=[],
    )
    cand_eval = {
        "summary": {
            "n_tasks": 2,
            "dev_low_rate": 1.0,
            "meta_pass_rate": None,
            "judge_pass_rate": 0.5,
            "in_loop_rate": 0.5,
            "dev_error_rate": 0.0,
            "avg_llm_calls": 1.0,
            "solved_ids_in_loop": ["t1"],
        },
        "records": [
            {
                "task_id": "t1",
                "family": "f1",
                "dev_low_status": "ok",
                "dev_low_ok": True,
                "dev_low_value": 10.0,
                "dev_low_truth": 10.0,
                "meta_checked": False,
                "meta_consistent": False,
                "meta_diag": "",
                "judge_checked": True,
                "judge_pass": True,
                "judge_diag": "LLM_JUDGE_PASS: ok",
                "code": "print(10.0)",
            },
            {
                "task_id": "t2",
                "family": "f2",
                "dev_low_status": "ok",
                "dev_low_ok": True,
                "dev_low_value": 20.0,
                "dev_low_truth": 20.0,
                "meta_checked": False,
                "meta_consistent": False,
                "meta_diag": "",
                "judge_checked": True,
                "judge_pass": False,
                "judge_diag": "LLM_JUDGE_REJECT: Uses EPSG:3857 shortcut.",
                "code": "g = g.to_crs('EPSG:3857')",
            },
        ],
    }
    dec = evaluate_gate("llm_judge", parent, cand_eval, soft_regress_delta=0.14)
    assert dec.accepted and dec.meta_blocked == ["t2"]
    digest = build_failure_digest(cand_eval, "llm_judge", minimal_prompt=True)
    assert "LLM_JUDGE_REJECT" in digest and "EPSG:3857" in digest


def test_extract_embedded_fences_in_multiline_string_and_smoke_import():
    """Embedded ``` lines inside a harness's multiline SYSTEM_PROMPT must not chop off top imports,
    and propose_mutation() must catch missing imports (NameError) before evaluation."""
    from meta_rsi.meta_agent import _extract_hypothesis_and_code, propose_mutation

    raw = (
        "thought\nI will include an example inside SYSTEM_PROMPT.\n<channel|>"
        "HYPOTHESIS: Preserve full module despite embedded fences.\n"
        "```python\n"
        "from pathlib import Path\n\n"
        'SYSTEM_PROMPT = """Return code inside:\n'
        "```python\n"
        "print(42)\n"
        "```\n"
        '"""\n\n'
        "def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict:\n"
        "    return {'code': None, 'trace': [], 'turns': 0}\n"
        "```\n"
    )
    hyp, code = _extract_hypothesis_and_code(raw)
    assert "Preserve full module" in hyp
    assert code is not None
    assert "from pathlib import Path" in code
    assert "SYSTEM_PROMPT" in code
    assert "def solve(" in code

    # Unquoted </think> or <channel|> inside a Python comment inside the final code block must NOT chop top imports!
    raw_comment_think = (
        "<|channel>thought\nDraft ```python\nx=1\n```\n<channel|>"
        "HYPOTHESIS: Strip reasoning tags cleanly.\n"
        "```python\n"
        "from pathlib import Path\n"
        "SYSTEM_PROMPT = 'Solve'\n"
        "def extract_code(text):\n"
        "    # Strip reasoning blocks ending in </think> or <channel|>\n"
        "    return text\n"
        "def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict:\n"
        "    return {'code': extract_code(SYSTEM_PROMPT), 'trace': [], 'turns': 1}\n"
        "```\n"
    )
    hyp_c, code_c = _extract_hypothesis_and_code(raw_comment_think)
    assert "Strip reasoning tags" in hyp_c
    assert code_c is not None
    assert "from pathlib import Path" in code_c
    assert "SYSTEM_PROMPT" in code_c
    assert "def extract_code(" in code_c
    assert "def solve(" in code_c

    # Smoke-import check in propose_mutation: first attempt omits Path (NameError), second succeeds.
    calls = {"n": 0}
    def flaky_meta(sys_p: str, usr_p: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                "HYPOTHESIS: broken import\n```python\n"
                "def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict:\n"
                "    return {'code': None, 'trace': [], 'turns': 0}\n```\n"
            )
        assert "NameError" in usr_p
        return (
            "HYPOTHESIS: fixed import\n```python\nfrom pathlib import Path\n"
            "def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict:\n"
            "    return {'code': None, 'trace': [], 'turns': 0}\n```\n"
        )

    with tempfile.TemporaryDirectory() as tmp:
        seed_p = Path(tmp) / "seed.py"
        seed_p.write_text("from pathlib import Path\ndef solve(a, b, c, d): return {'code': None, 'trace': [], 'turns': 0}\n")
        parent = ArchiveEntry(0, None, str(seed_p), "seed", 0.5, 0.5, 0.0, 1.0, [])
        ev = {
            "summary": {"n_tasks": 1, "dev_low_rate": 0.5, "meta_pass_rate": 0.5, "in_loop_rate": 0.5, "dev_error_rate": 0.0, "avg_llm_calls": 1.0},
            "records": [],
        }
        hyp2, code2 = propose_mutation(LLMClient("mock", "m", mock_fn=flaky_meta), parent, ev, [], "meta_soft")
        assert calls["n"] == 2
        assert "from pathlib import Path" in code2


def test_geoanalyst_audit_and_probe_loader(tmp_path):
    """GeoAnalystBench static geospatial soundness audit and probe-dir loader work as expected."""
    from meta_rsi.eval.geoanalyst_eval import audit_geo_script
    from meta_rsi.training.collect_trajectories import load_probes_from_dir

    vec_task = {"needs_raster": False, "needs_vector_metric": True}
    rast_task = {"needs_raster": True, "needs_vector_metric": False}

    # Distorted EPSG:3857 should fail
    bad_wm = "import geopandas as gpd\ng = gpd.read_file('a.geojson').to_crs('EPSG:3857')\nprint(g.geometry.area.sum())\n"
    a_wm = audit_geo_script(bad_wm, vec_task)
    assert a_wm["syntax_ok"] and a_wm["uses_3857"] and not a_wm["strict_geo_sound"]

    # Degree-space buffer without projection should fail
    bad_deg = "import geopandas as gpd\ng = gpd.read_file('a.geojson')\nb = g.buffer(500)\nprint(b.area.sum())\n"
    a_deg = audit_geo_script(bad_deg, vec_task)
    assert a_deg["syntax_ok"] and not a_deg["no_degree_metric"] and not a_deg["strict_geo_sound"]

    # Proper local UTM / AEQD projection should pass
    good_utm = "import geopandas as gpd\ng = gpd.read_file('a.geojson')\ng = g.to_crs(g.estimate_utm_crs())\nprint(g.buffer(500).area.sum())\n"
    a_good = audit_geo_script(good_utm, vec_task)
    assert a_good["strict_geo_sound"]

    # Raster without nodata handling fails; with masked=True passes
    bad_r = "import rasterio\nwith rasterio.open('dem.tif') as s:\n    arr = s.read(1)\n    print(arr.mean())\n"
    good_r = "import rasterio\nwith rasterio.open('dem.tif') as s:\n    arr = s.read(1, masked=True)\n    print(arr.mean())\n"
    assert not audit_geo_script(bad_r, rast_task)["strict_geo_sound"]
    assert audit_geo_script(good_r, rast_task)["strict_geo_sound"]

    # load_probes_from_dir loads valid .py probes cleanly
    pdir = tmp_path / "probes"
    pdir.mkdir()
    (pdir / "p1.py").write_text("from pathlib import Path\ndef perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:\n    return {'mode': 'noop'}\n")
    loaded = load_probes_from_dir(pdir, tmp_path / "cache")
    assert loaded is not None and len(loaded) == 1 and loaded[0][0] == "p1"






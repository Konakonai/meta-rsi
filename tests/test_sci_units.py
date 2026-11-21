"""Unit tests for SciUnits non-GIS invariance benchmark and tabular metamorphic perturbation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.gsb.harness import matches
from benchmarks.sci_units.audit import audit_dataset
from benchmarks.sci_units.build import make_task, read_workspace_si
from benchmarks.sci_units.families import BY_NAME, FAMILIES
from meta_rsi.metamorphic import perturb_workspace


def test_sci_units_build_separability_and_metamorphic_invariance():
    """All 12 SciUnits families preserve truth across all 4 variants + metamorphic perturbations,
    while naive SI/dropna solvers pass `low` (>95%) and fail oracle variants (0% on non-control)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tasks = []
        for idx, fam in enumerate(FAMILIES):
            rng = np.random.default_rng(np.random.SeedSequence([7, idx]))
            t = make_task(fam, rng, root, f"{fam.name}-test-0000", "test")
            tasks.append(t)

            # Verify label-free metamorphic perturbation preserves physical ground truth!
            low_dir = root / t["variants"]["low"]["dir"]
            for flip in (False, True):
                pdir = root / f"pert_{fam.name}_{int(flip)}"
                meta = perturb_workspace(low_dir, pdir, flip_gpkg_crs=flip, seed=42 + int(flip))
                assert meta["mode"] == "tabular_unit_cal_sentinel_shuffle"
                val = float(fam.solve(read_workspace_si(pdir), t["params"]))
                assert matches(val, t["variants"]["low"]["truth"], t["answer"]), (
                    f"Metamorphic drift on {fam.name} (flip={flip}): got={val}, truth={t['variants']['low']['truth']}"
                )

        (root / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
        audit = audit_dataset(root)
        assert audit["ref_pass"] == len(FAMILIES)
        assert audit["non_control_naive_low_pass"] == audit["non_control_total"]
        assert audit["non_control_naive_all_pass"] == 0


def test_sci_units_self_certification_gate():
    """Verify zero-label algebraic conservation certification works outside GIS on SciUnits."""
    from meta_rsi.verifier_agent import certify_perturbation_operator

    fam = BY_NAME["ideal_gas_moles_filtered"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(123)
        t = make_task(fam, rng, root, "ideal_gas-test-0000", "test")
        low_dir = root / t["variants"]["low"]["dir"]
        naive_script = (
            "import pandas as pd\n"
            "df = pd.read_csv('vessels.csv').dropna()\n"
            f"sub = df[df['gas_type'] == {t['params']['gas_type']!r}]\n"
            "print(((sub['pressure'] * sub['volume']) / (8.314462618 * sub['temperature'])).sum())\n"
        )
        passing_cases = [(low_dir, naive_script, float(t["variants"]["low"]["truth"]), "float")]

        good_probe = '''
import json
from pathlib import Path
import pandas as pd

def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src_dir / "metadata.json").read_text())
    df = pd.read_csv(src_dir / "vessels.csv")
    df["temperature"] = df["temperature"] - 273.15
    df["pressure"] = df["pressure"] / 1000.0
    meta["tables"]["vessels.csv"]["units"]["temperature"] = "degC"
    meta["tables"]["vessels.csv"]["units"]["pressure"] = "kPa"
    df.to_csv(dst_dir / "vessels.csv", index=False)
    (dst_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"mode": "degC_kPa_unit_swap"}
'''
        res_ok = certify_perturbation_operator(good_probe, [low_dir], passing_cases)
        assert res_ok["certified"], f"Expected valid SciUnits probe certified, got: {res_ok}"
        assert res_ok["metrics"]["caught"] == 1

        bad_probe = '''
import json
from pathlib import Path
import pandas as pd

def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src_dir / "metadata.json").read_text())
    df = pd.read_csv(src_dir / "vessels.csv")
    # Corrupted: changes values by 2x without updating metadata!
    df["pressure"] = df["pressure"] * 2.0
    df.to_csv(dst_dir / "vessels.csv", index=False)
    (dst_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"mode": "corrupt_2x"}
'''
        res_bad = certify_perturbation_operator(bad_probe, [low_dir], passing_cases)
        assert not res_bad["certified"] and res_bad["stage"] == "conservation"


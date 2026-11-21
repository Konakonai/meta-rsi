"""Build SciUnits non-GIS benchmark (`data/sci_v1`) with verified physical invariance.

Usage:
    python -m benchmarks.sci_units.build --out data/sci_v1 --train 60 --test 20 --heldout 30 --seed 0
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .families import FAMILIES, SciFamily, SciSpec, VARIANTS_SCI
from .units import from_si, pick_alt_unit, to_si


def read_workspace_si(d: Path) -> dict[str, pd.DataFrame]:
    """Reference metadata-aware loader: applies missing_sentinel, linear calibration, and unit -> SI."""
    meta = json.loads((d / "metadata.json").read_text())
    out: dict[str, pd.DataFrame] = {}
    for fname, tmeta in meta["tables"].items():
        stem = Path(fname).stem
        df = pd.read_csv(d / fname)
        sent = tmeta.get("missing_sentinel")
        if sent is not None:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mask = np.isclose(df[col].to_numpy(dtype=float), float(sent), atol=1e-6, equal_nan=False)
                    if np.any(mask):
                        df.loc[mask, col] = np.nan
        cals = tmeta.get("calibration") or {}
        for col, cfg in cals.items():
            if col in df.columns:
                sc = float(cfg.get("scale", 1.0))
                off = float(cfg.get("offset", 0.0))
                df[col] = df[col].astype(float) * sc + off
        units = tmeta.get("units") or {}
        for col, unit in units.items():
            if col in df.columns:
                df[col] = to_si(df[col].astype(float), unit)
        out[stem] = df
    return out


def write_variant(spec: SciSpec, variant: str, out: Path, rng: np.random.Generator) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    meta_tables: dict[str, dict] = {}
    files: dict[str, str] = {}

    v_idx = VARIANTS_SCI.index(variant)
    for stem, tspec in spec.tables.items():
        fname = f"{stem}.csv"
        files[stem] = fname
        df = tspec.df.copy()

        # 1. Choose units
        units_map: dict[str, str] = {}
        for col, si_u in tspec.si_units.items():
            if variant == "low":
                u = si_u
            else:
                u = pick_alt_unit(si_u, rng, variant_idx=v_idx)
            units_map[col] = u
            if u != si_u:
                df[col] = from_si(df[col].to_numpy(dtype=float), u)

        # Optional constant time-origin offset on non-low variants (invariant for dt integrals)
        if tspec.time_col and variant in ("sentinel_shifted", "permuted_mixed"):
            shift_val = float(rng.choice([120.0, 500.0, 1800.0]))
            df[tspec.time_col] = df[tspec.time_col].astype(float) + shift_val

        # 2. Optional affine calibration packing: physical = raw * scale + offset => raw = (physical - offset) / scale
        cal_map: dict[str, dict[str, float]] = {}
        for col in tspec.calibratable:
            if variant in ("low", "unit_imperial"):
                sc, off = 1.0, 0.0
            else:
                sc = float(rng.choice([0.1, 0.25, 2.0, 4.0]))
                off = float(rng.choice([-15.0, 12.0, 30.0]))
                df[col] = (df[col].astype(float) - off) / sc
            cal_map[col] = {"scale": sc, "offset": off}

        # 3. Missing-value sentinel encoding
        sentinel: float | None = None
        if variant == "sentinel_shifted" and tspec.nullable_cols:
            sentinel = -999.0
        elif variant == "permuted_mixed" and tspec.nullable_cols:
            sentinel = -9999.0
        if sentinel is not None:
            for col in tspec.nullable_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(sentinel)

        # 4. Auxiliary unused NaN column on permuted_mixed (catches indiscriminate df.dropna())
        if variant == "permuted_mixed":
            df["aux_sensor_note"] = [np.nan if m else "ok" for m in (rng.random(len(df)) < 0.72)]

        # 5. Row & column permutation on all non-low variants
        if variant != "low":
            if len(df) > 1:
                df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
            cols = list(df.columns)
            rng.shuffle(cols)
            df = df[cols]

        # Round floats cleanly while keeping >=8 significant digits
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].round(8)
        df.to_csv(out / fname, index=False)

        meta_tables[fname] = {
            "units": units_map,
            "calibration": cal_map,
            "missing_sentinel": sentinel,
        }

    meta_obj = {
        "schema_version": "1.0",
        "calibration_formula": "value_in_declared_unit = raw_csv_value * scale + offset",
        "tables": meta_tables,
    }
    (out / "metadata.json").write_text(json.dumps(meta_obj, indent=2))
    files["metadata"] = "metadata.json"
    return files


def format_prompt(spec: SciSpec, question: str) -> str:
    lines = ["The current directory contains these files:"]
    for stem, tspec in spec.tables.items():
        lines.append(f"- `{stem}.csv`: {tspec.description}.")
    lines.append("- `metadata.json`: JSON file documenting table metadata (column units, calibration, and missing-value sentinels).")
    lines.append(f"Write a Python script that {question}. Print only the final number.")
    return "\n".join(lines)


def make_task(fam: SciFamily, rng: np.random.Generator, root: Path, task_id: str, split: str) -> dict:
    task_dir = root / task_id
    for _ in range(20):
        if task_dir.exists():
            shutil.rmtree(task_dir)
        spec = fam.build(rng)
        variants: dict[str, dict] = {}
        files: dict[str, str] = {}
        ok = True
        low_truth: float | None = None
        for v in VARIANTS_SCI:
            d = task_dir / v
            files = write_variant(spec, v, d, rng)
            si_tables = read_workspace_si(d)
            truth = float(fam.solve(si_tables, spec.params))
            if not np.isfinite(truth) or truth <= 1e-6:
                ok = False
                break
            if v == "low":
                low_truth = truth
            else:
                assert low_truth is not None
                if abs(truth - low_truth) / max(abs(low_truth), 1e-12) > 1e-5:
                    ok = False
                    break
            variants[v] = {"dir": f"{task_id}/{v}", "truth": truth}
        if ok and low_truth is not None:
            return {
                "task_id": task_id,
                "family": fam.name,
                "group": fam.group,
                "split": split,
                "answer": fam.answer,
                "params": spec.params,
                "files": files,
                "prompt": format_prompt(spec, fam.question(spec.params)),
                "variants": variants,
            }
    raise RuntimeError(f"Failed to generate invariant SciUnits task for {fam.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SciUnits non-GIS invariance benchmark")
    ap.add_argument("--out", default="data/sci_v1")
    ap.add_argument("--train", type=int, default=60, help="train instances per seen family")
    ap.add_argument("--test", type=int, default=20, help="test instances per seen family")
    ap.add_argument("--heldout", type=int, default=30, help="test instances per held-out family")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    tasks: list[dict] = []
    for idx, fam in enumerate(FAMILIES):
        fam_rng = np.random.default_rng(np.random.SeedSequence([a.seed, idx]))
        plan = [("test", a.heldout)] if fam.held_out else [("train", a.train), ("test", a.test)]
        for split, n in plan:
            for i in range(n):
                tasks.append(make_task(fam, fam_rng, root, f"{fam.name}-{split}-{i:04d}", split))
        print(f"  built {fam.name:30s} ({'heldout' if fam.held_out else 'seen':7s}, group={fam.group})", flush=True)

    with open(root / "tasks.jsonl", "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    print(Counter((t["group"], t["split"]) for t in tasks))
    print(f"Total {len(tasks)} tasks written to {root}")


if __name__ == "__main__":
    main()

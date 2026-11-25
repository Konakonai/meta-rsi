"""Build held-out failure-mechanism variants for GSB v2 test split (E2 in docs/NEXT_EXPERIMENTS.md).

Evaluates whether gains on GSB v2 generalize beyond the specific mechanisms exercised by the
metamorphic probes (Web Mercator, UTM limits, Mercator/TM storage, km units). Every variant keeps
the task at its original `low` location so ONLY the storage/encoding mechanism changes:

  Near transfer (mechanism family touched by probes, in an unseen form):
    - `us_ft`     : every `.gpkg` vector layer stored in a local Transverse Mercator CRS with `+units=us-ft`
    - `albers`    : every `.gpkg` vector layer stored in a local Albers Equal-Area Conic CRS (`+units=m`)
  Far transfer (mechanism untouched by any probe):
    - `nodata_nan`: every `.tif` GeoTIFF stored as float32 with IEEE-754 `NaN` nodata instead of integer sentinel
    - `csv_lon360`: every `.csv` point layer with longitudes converted to `[0, 360)` (west-of-Greenwich only,
                    so at least one longitude value actually changes)

Reference truths are recomputed from the written files with each family's reference solver; any
variant whose reference truth moves from `low` by more than build.py's invariance tolerance
(0.8% float, exact count) or is non-finite is kept with `scored: false`.

Usage:
    nix-shell --run "python -m benchmarks.gsb_v2.build_heldout_mech"   # -> data/v2_heldout_mech/
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS

from . import build as B
from .families import BY_NAME
from .geo import reference_aeqd_crs

MECHANISMS = (
    ("us_ft", "near"),
    ("albers", "near"),
    ("nodata_nan", "far"),
    ("csv_lon360", "far"),
)


def _us_ft_crs(lat: float, lon: float) -> CRS:
    return CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat:.6f} +lon_0={lon:.6f} +k=0.9996 "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=us-ft +no_defs"
    )


def _local_albers_crs(lat: float, lon: float) -> CRS:
    """Non-degenerate local Albers Equal-Area Conic (guarantees lat_1 + lat_2 != 0 even at equator)."""
    if lat >= 0.0:
        lat_1, lat_2 = lat + 0.5, lat + 2.5
    else:
        lat_1, lat_2 = lat - 2.5, lat - 0.5
    return CRS.from_proj4(
        f"+proj=aea +lat_0={lat:.6f} +lon_0={lon:.6f} "
        f"+lat_1={lat_1:.6f} +lat_2={lat_2:.6f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def _solve_dir(fam, params: dict, files: dict[str, str], d: Path) -> float:
    layers, rasters = B.read_workspace(d, files)
    if fam.large:
        return float(fam.solve(layers, rasters, None, None, params))
    crs_obj = reference_aeqd_crs(layers, rasters)
    return float(fam.solve(layers, rasters, lambda g, crs=crs_obj: g.to_crs(crs), crs_obj, params))


def _write_one_mechanism(
    task: dict,
    low_dir: Path,
    out_dir: Path,
    mech: str,
) -> dict | None:
    """Apply one held-out mechanism to `low_dir` -> `out_dir`. Return metadata dict, or None if N/A."""
    files = task["files"]
    low = task["variants"]["low"]
    lat, lon = float(low["lat"]), float(low["lon"])

    has_gpkg = any(fn.endswith(".gpkg") for fn in files.values())
    has_tif = any(fn.endswith(".tif") for fn in files.values())
    has_csv = any(fn.endswith(".csv") for fn in files.values())

    if mech in ("us_ft", "albers") and not has_gpkg:
        return None
    if mech == "nodata_nan" and not has_tif:
        return None
    if mech == "csv_lon360":
        if not has_csv:
            return None
        # Only apply when at least one CSV point lies west of Greenwich (lon < 0) so values truly change.
        any_west = False
        for fn in files.values():
            if fn.endswith(".csv"):
                df = pd.read_csv(low_dir / fn)
                if "lon" in df.columns and bool((df["lon"] < 0.0).any()):
                    any_west = True
                    break
        if not any_west:
            return None

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage: dict[str, str] = {}

    if mech in ("us_ft", "albers"):
        dst_crs = _us_ft_crs(lat, lon) if mech == "us_ft" else _local_albers_crs(lat, lon)
        for stem, fn in files.items():
            src_p, dst_p = low_dir / fn, out_dir / fn
            if fn.endswith(".gpkg"):
                g = gpd.read_file(src_p)
                g.to_crs(dst_crs).to_file(dst_p, driver="GPKG")
                storage[stem] = dst_crs.to_proj4()
            else:
                shutil.copy2(src_p, dst_p)
                storage[stem] = "unchanged"

    elif mech == "nodata_nan":
        for stem, fn in files.items():
            src_p, dst_p = low_dir / fn, out_dir / fn
            if fn.endswith(".tif"):
                with rasterio.open(src_p) as src:
                    arr = src.read(1).astype(np.float32)
                    if src.nodata is not None and np.isfinite(src.nodata):
                        arr[np.isclose(arr, float(src.nodata))] = np.float32(np.nan)
                    prof = src.profile.copy()
                    prof.update(dtype="float32", nodata=float("nan"))
                    with rasterio.open(dst_p, "w", **prof) as dst:
                        dst.write(arr, 1)
                storage[stem] = "float32 nodata=NaN"
            else:
                shutil.copy2(src_p, dst_p)
                storage[stem] = "unchanged"

    elif mech == "csv_lon360":
        for stem, fn in files.items():
            src_p, dst_p = low_dir / fn, out_dir / fn
            if fn.endswith(".csv"):
                df = pd.read_csv(src_p)
                df["lon"] = np.where(df["lon"] < 0.0, df["lon"] + 360.0, df["lon"]).round(7)
                df.to_csv(dst_p, index=False)
                storage[stem] = "EPSG:4326 (csv lon 0-360)"
            else:
                shutil.copy2(src_p, dst_p)
                storage[stem] = "unchanged"
    else:
        raise ValueError(mech)

    return {"storage": storage}


def _task_worker(args: tuple[dict, str, str]) -> dict:
    task, v2_root_s, out_root_s = args
    v2_root, out_root = Path(v2_root_s), Path(out_root_s)
    fam = BY_NAME[task["family"]]
    tid = task["task_id"]
    low = task["variants"]["low"]
    low_dir = v2_root / low["dir"]
    low_truth = float(low["truth"])

    variants: dict[str, dict] = {}
    for mech, transfer in MECHANISMS:
        vdir = out_root / tid / mech
        info = _write_one_mechanism(task, low_dir, vdir, mech)
        if info is None:
            continue
        try:
            truth = _solve_dir(fam, task["params"], task["files"], vdir)
            finite = bool(np.isfinite(truth))
        except Exception:
            truth = float("nan")
            finite = False
        rel = abs(truth - low_truth) / max(abs(low_truth), 1e-12) if finite else float("inf")
        invariant = bool(finite and rel <= (0.0 if task["answer"] == "count" else 0.008))
        variants[mech] = {
            "dir": f"{tid}/{mech}",
            "lat": low["lat"],
            "lon": low["lon"],
            "mechanism": mech,
            "transfer": transfer,
            "truth": truth if finite else None,
            "finite": finite,
            "rel_diff_vs_low": round(rel, 8) if np.isfinite(rel) else None,
            "invariant": invariant,
            "scored": bool(finite and invariant),
            **info,
        }

    return {
        "task_id": tid,
        "family": task["family"],
        "answer": task["answer"],
        "large": fam.large,
        "has_raster": fam.has_raster,
        "held_out": fam.held_out,
        "files": task["files"],
        "store_proj_gpkg": task.get("store_proj_gpkg", False),
        "low": {"lat": low["lat"], "lon": low["lon"], "truth": low_truth},
        "variants": variants,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v2", default="data/v2")
    ap.add_argument("--out", default="data/v2_heldout_mech")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    v2_root = Path(a.v2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    test_tasks = [t for t in map(json.loads, open(v2_root / "tasks.jsonl")) if t["split"] == "test"]
    jobs = [(t, str(v2_root), str(out)) for t in test_tasks]
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
        built = list(ex.map(_task_worker, jobs))

    with open(out / "tasks.jsonl", "w") as fh:
        for t in built:
            fh.write(json.dumps(t) + "\n")

    vs = [v for t in built for v in t["variants"].values()]
    by_mech = Counter(v["mechanism"] for v in vs)
    scored_by_mech = Counter(v["mechanism"] for v in vs if v["scored"])
    not_scored = Counter(f"{t['family']}|{v['mechanism']}" for t in built for v in t["variants"].values() if not v["scored"])
    meta = {
        "n_tasks": len(built),
        "n_tasks_with_any_variant": sum(bool(t["variants"]) for t in built),
        "n_variants": len(vs),
        "scored": sum(v["scored"] for v in vs),
        "by_mechanism": dict(by_mech),
        "scored_by_mechanism": dict(scored_by_mech),
        "not_scored_by_family_mechanism": dict(not_scored),
    }
    (out / "heldout_mech_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

"""Build GSB v2 dataset (`data/v2`) with verified inter-variant physical invariance.

Usage:
    python -m benchmarks.gsb_v2.build --out data/v2 --train 100 --test 20 --heldout 30 --seed 0
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity

from benchmarks.gsb.harness import matches

from .families import FAMILIES, Family, Spec
from .geo import (
    VARIANTS_LARGE,
    VARIANTS_LOCAL,
    VARIANTS_RASTER,
    place,
    reference_aeqd_crs,
    sample_center,
    sample_center_large,
    storage_crs_for_raster,
    write_physical_geotiff,
)

TYPE_WORDS = {"point": "point", "line": "line", "polygon": "polygon"}


def read_layer(path: Path) -> gpd.GeoDataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return gpd.GeoDataFrame(
            df.drop(columns=["lon", "lat"]),
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )
    return gpd.read_file(path)


def read_workspace(d: Path, files: dict[str, str]) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, Path]]:
    layers: dict[str, gpd.GeoDataFrame] = {}
    rasters: dict[str, Path] = {}
    for stem, fname in files.items():
        p = d / fname
        if fname.endswith(".tif"):
            rasters[stem] = p
        else:
            layers[stem] = read_layer(p)
    return layers, rasters


def choose_formats(spec: Spec, rng: np.random.Generator) -> dict[str, str]:
    files: dict[str, str] = {}
    for stem, kind in spec.kinds.items():
        opts = ["geojson", "gpkg", "csv"] if kind == "point" else ["geojson", "gpkg"]
        files[stem] = f"{stem}.{rng.choice(opts)}"
    for stem in spec.rasters:
        files[stem] = f"{stem}.tif"
    return files


def write_variant(
    spec: Spec,
    files: dict[str, str],
    variant: str,
    lat: float,
    lon: float,
    out: Path,
    *,
    store_proj_gpkg: bool,
    large: bool,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stem, gdf in spec.layers.items():
        if large:
            geoms = [shapely.affinity.translate(x, lon, lat) for x in gdf.geometry]
        else:
            geoms = place(list(gdf.geometry), lat, lon)
        g = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=geoms, crs="EPSG:4326")
        fname = files[stem]
        if fname.endswith(".csv"):
            df = pd.DataFrame(g.drop(columns="geometry"))
            df["lon"], df["lat"] = g.geometry.x.round(7), g.geometry.y.round(7)
            df.to_csv(out / fname, index=False)
        elif fname.endswith(".gpkg"):
            if store_proj_gpkg and not large and variant != "antimeridian":
                if variant == "polar":
                    dst = "EPSG:3413" if lat >= 0 else "EPSG:3031"
                else:
                    dst = g.estimate_utm_crs()
                g.to_crs(dst).to_file(out / fname, driver="GPKG")
            else:
                g.to_file(out / fname, driver="GPKG")
        else:
            g.to_file(out / fname, driver="GeoJSON")

    for stem, rspec in spec.rasters.items():
        dst_crs, nodata = storage_crs_for_raster(variant, lat, force_4326_high=rspec.force_4326_high)
        write_physical_geotiff(
            out / files[stem],
            lat=lat,
            lon=lon,
            extent_m=rspec.extent_m,
            pixel_m=rspec.pixel_m,
            field_fn=rspec.field_fn,
            valid_fn=rspec.valid_fn,
            dst_crs=dst_crs,
            nodata=nodata,
        )


def format_prompt(spec: Spec, files: dict[str, str], question: str) -> str:
    lines = ["The current directory contains these files:"]
    for stem, fname in files.items():
        if fname.endswith(".tif"):
            desc = spec.rasters[stem].description
            lines.append(f"- `{fname}`: single-band GeoTIFF raster representing {desc} (may contain nodata pixels).")
        else:
            cols = ", ".join(f"`{c}` ({t})" for c, t in spec.columns.get(stem, []))
            if fname.endswith(".csv"):
                lines.append(
                    f"- `{fname}`: {TYPE_WORDS[spec.kinds[stem]]} features as CSV with columns {cols}, "
                    "`lon`, `lat` (WGS84 degrees)."
                )
            else:
                lines.append(f"- `{fname}`: {TYPE_WORDS[spec.kinds[stem]]} layer with columns {cols}.")
    lines.append(f"Write a Python script that {question}. Print only the final number.")
    return "\n".join(lines)


def make_task(fam: Family, rng: np.random.Generator, root: Path, task_id: str, split: str) -> dict:
    variant_names = VARIANTS_LARGE if fam.large else (VARIANTS_RASTER if fam.has_raster else VARIANTS_LOCAL)
    task_dir = root / task_id

    for attempt in range(30):
        if task_dir.exists():
            shutil.rmtree(task_dir)
        spec = fam.build(rng)
        files = choose_formats(spec, rng)
        store_proj_gpkg = bool(rng.random() < 0.30) and not fam.large
        variants: dict[str, dict] = {}
        ok_task = True
        low_truth: float | None = None

        for v in variant_names:
            lat, lon = (sample_center_large if fam.large else sample_center)(v, rng)
            d = task_dir / v
            write_variant(spec, files, v, lat, lon, d, store_proj_gpkg=store_proj_gpkg, large=fam.large)
            layers, rasters = read_workspace(d, files)
            if fam.large:
                crs_obj = None
                proj = None
            else:
                crs_obj = reference_aeqd_crs(layers, rasters)
                proj = lambda g, crs=crs_obj: g.to_crs(crs)
            truth = float(fam.solve(layers, rasters, proj, crs_obj, spec.params))
            if not np.isfinite(truth) or truth <= 1e-6:
                ok_task = False
                break
            if v == "low":
                low_truth = truth
            elif not fam.large:
                assert low_truth is not None
                # Ensure discrete/ellipsoidal inter-variant invariance holds tightly (<0.8%)
                rel_diff = abs(truth - low_truth) / max(abs(low_truth), 1e-12)
                tol = 0.0 if fam.answer == "count" else 0.008
                if rel_diff > tol:
                    ok_task = False
                    break
            variants[v] = {"dir": f"{task_id}/{v}", "lat": round(lat, 5), "lon": round(lon, 5), "truth": truth}

        if ok_task and low_truth is not None:
            return {
                "task_id": task_id,
                "family": fam.name,
                "group": fam.group,
                "split": split,
                "answer": fam.answer,
                "has_raster": fam.has_raster,
                "large": fam.large,
                "params": spec.params,
                "files": files,
                "store_proj_gpkg": store_proj_gpkg,
                "prompt": format_prompt(spec, files, fam.question(spec.params)),
                "variants": variants,
            }

    raise RuntimeError(f"Failed to generate physically invariant task for family={fam.name} after 30 attempts")


def _build_family_worker(args: tuple[int, str, int, int, int, int]) -> list[dict]:
    fam_idx, root_str, seed, n_train, n_test, n_heldout = args
    fam = FAMILIES[fam_idx]
    root = Path(root_str)
    fam_rng = np.random.default_rng(np.random.SeedSequence([seed, fam_idx]))
    plan = [("test", n_heldout)] if fam.held_out else [("train", n_train), ("test", n_test)]
    out_tasks = []
    for split, n in plan:
        for i in range(n):
            out_tasks.append(make_task(fam, fam_rng, root, f"{fam.name}-{split}-{i:04d}", split))
    print(
        f"  built {fam.name:28s} ({'heldout' if fam.held_out else 'seen':7s}, group={fam.group}, n={len(out_tasks)})",
        flush=True,
    )
    return out_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GSB v2 multi-modal benchmark dataset")
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--train", type=int, default=100, help="train instances per seen family")
    ap.add_argument("--test", type=int, default=20, help="test instances per seen family")
    ap.add_argument("--heldout", type=int, default=30, help="test instances per held-out family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    from concurrent.futures import ProcessPoolExecutor

    jobs = [(idx, str(root), a.seed, a.train, a.test, a.heldout) for idx in range(len(FAMILIES))]
    tasks: list[dict] = []
    if a.workers <= 1:
        for j in jobs:
            tasks.extend(_build_family_worker(j))
    else:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(FAMILIES))) as ex:
            for fam_tasks in ex.map(_build_family_worker, jobs):
                tasks.extend(fam_tasks)

    with open(root / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(Counter((t["group"], t["split"]) for t in tasks))
    print(f"Total {len(tasks)} tasks written to {root}")


if __name__ == "__main__":
    main()

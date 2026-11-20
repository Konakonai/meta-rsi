"""Generate task instances: files for three location variants, prompt, and ground truth.

    python -m benchmarks.gsb.build --out data/v1 --train 150 --test 20 --heldout 40 --seed 0
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

from .families import FAMILIES, Spec
import shapely.affinity

from .geo import LARGE_VARIANTS, VARIANTS, local_crs, place, sample_center, sample_center_large

TYPE_WORDS = {"point": "point", "line": "line", "polygon": "polygon"}


def read_layer(path: Path) -> gpd.GeoDataFrame:
    """Read a layer the way a straightforward script would (CSV -> WGS84 points)."""
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return gpd.GeoDataFrame(df.drop(columns=["lon", "lat"]),
                                geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    return gpd.read_file(path)


def read_layers(d: Path, files: dict) -> dict:
    return {stem: read_layer(d / fname) for stem, fname in files.items()}


def reference_proj(layers: dict):
    """Metric CRS for ground truth: azimuthal equidistant at the data center."""
    allg = pd.concat([g.to_crs("EPSG:4326").geometry for g in layers.values()])
    minx, miny, maxx, maxy = allg.total_bounds
    crs = local_crs((miny + maxy) / 2, (minx + maxx) / 2)
    return lambda g: g.to_crs(crs)


def choose_formats(spec: Spec, rng) -> dict:
    files = {}
    for stem, kind in spec.kinds.items():
        opts = ["geojson", "gpkg", "csv"] if kind == "point" else ["geojson", "gpkg"]
        files[stem] = f"{stem}.{rng.choice(opts)}"
    return files


def write_variant(spec: Spec, files: dict, lat: float, lon: float, out: Path, store_utm: bool, large: bool):
    out.mkdir(parents=True, exist_ok=True)
    for stem, gdf in spec.layers.items():
        if large:  # geometry is in degrees around (0, 0): shift it
            geoms = [shapely.affinity.translate(x, lon, lat) for x in gdf.geometry]
        else:  # geometry is in local metres: place it with an azimuthal equidistant projection
            geoms = place(list(gdf.geometry), lat, lon)
        g = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=geoms, crs="EPSG:4326")
        fname = files[stem]
        if fname.endswith(".csv"):
            df = pd.DataFrame(g.drop(columns="geometry"))
            df["lon"], df["lat"] = g.geometry.x.round(7), g.geometry.y.round(7)
            df.to_csv(out / fname, index=False)
        elif fname.endswith(".gpkg"):
            (g.to_crs(g.estimate_utm_crs()) if store_utm else g).to_file(out / fname, driver="GPKG")
        else:
            g.to_file(out / fname, driver="GeoJSON")


def prompt(spec: Spec, files: dict, question: str) -> str:
    lines = ["The current directory contains these files:"]
    for stem, fname in files.items():
        cols = ", ".join(f"`{c}` ({t})" for c, t in spec.columns.get(stem, []))
        if fname.endswith(".csv"):
            lines.append(f"- `{fname}`: {TYPE_WORDS[spec.kinds[stem]]} features as CSV with columns {cols}, "
                         f"`lon`, `lat` (WGS84 degrees).")
        else:
            lines.append(f"- `{fname}`: {TYPE_WORDS[spec.kinds[stem]]} layer with columns {cols}.")
    lines.append(f"Write a Python script that {question}. Print only the final number.")
    return "\n".join(lines)


def make_task(fam, rng, root: Path, task_id: str, split: str) -> dict:
    spec = fam.build(rng)
    files = choose_formats(spec, rng)
    store_utm = bool(rng.random() < 0.25) and not fam.large
    variants = {}
    for v in (LARGE_VARIANTS if fam.large else VARIANTS):
        lat, lon = (sample_center_large if fam.large else sample_center)(v, rng)
        d = root / task_id / v
        write_variant(spec, files, lat, lon, d, store_utm, fam.large)
        layers = read_layers(d, files)
        truth = fam.solve(layers, None if fam.large else reference_proj(layers), spec.params)
        variants[v] = {"dir": f"{task_id}/{v}", "lat": lat, "lon": lon, "truth": float(truth)}
    return {"task_id": task_id, "family": fam.name, "split": split, "answer": fam.answer,
            "params": spec.params, "files": files, "store_utm": store_utm,
            "prompt": prompt(spec, files, fam.question(spec.params)), "variants": variants}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1")
    ap.add_argument("--train", type=int, default=150, help="train instances per seen family")
    ap.add_argument("--test", type=int, default=20, help="test instances per seen family")
    ap.add_argument("--heldout", type=int, default=40, help="test instances per held-out family")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(a.seed)
    tasks = []
    for fam in FAMILIES:
        plan = [("test", a.heldout)] if fam.held_out else [("train", a.train), ("test", a.test)]
        for split, n in plan:
            for i in range(n):
                tasks.append(make_task(fam, rng, root, f"{fam.name}-{split}-{i:04d}", split))
    with open(root / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(Counter((t["family"], t["split"]) for t in tasks))
    print(len(tasks), "tasks written to", root)


if __name__ == "__main__":
    main()

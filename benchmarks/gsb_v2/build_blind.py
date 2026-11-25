"""Blind relocation variants for the GSB v2 test split (review item M1; protocol in docs/BLIND_PROTOCOL.md).

The designed variants (`high`, `polar`, `antimeridian`, `zone_edge`) sit in the same regimes as the gate's probes
(58.3N / 56.2S / 86N). Blind variants put the same task content at locations drawn from a fixed, probe-agnostic
distribution, so a script written on `low` can be scored where nobody chose the location:

  local families  8 centres per task, 2 per |latitude| band [0,30) [30,60) [60,80) [80,89], area-uniform within
                  the band, random hemisphere, uniform longitude (raster scenes are kept off the antimeridian)
  large families  8 centres per task, area-uniform latitude and uniform longitude over the range that keeps the
                  whole scene inside +-80 deg latitude and +-180 deg longitude

Storage follows the benchmark's own conventions, chosen by latitude instead of by variant name: GeoJSON / CSV in
WGS84; projected GeoPackages in UTM (polar stereographic beyond 84N / 80S, WGS84 if the scene crosses 180 deg);
GeoTIFFs as `low` (EPSG:3857, nodata -9999) within 2.2 deg of the equator, as `high` up to 85 deg, as `polar`
beyond. Truths are recomputed with each family's reference solver on the written files. A local variant whose
truth moves from the `low` truth by more than build.py's invariance tolerance (0.8%, counts exact) is kept
but marked `scored: false`.

The task content is recovered by replaying the seeded `data/v2` build (every replayed task is checked against
the stored one), so no task text or data changes; the blind locations use their own seed.

    nix-shell --run "python -m benchmarks.gsb_v2.build_blind"   # -> data/v2_blind/
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity
from pyproj import CRS

from . import build as B
from .families import FAMILIES
from .geo import place, reference_aeqd_crs, storage_crs_for_raster, write_physical_geotiff

BANDS = ((0.0, 30.0), (30.0, 60.0), (60.0, 80.0), (80.0, 89.0))
PER_BAND = 2
# Centres the gate relocates data to: built-in shift_and_shuffle, crs_storage_flip, certified probe_r2_polar_laea
# (which keeps the source longitude). Used only to describe each blind location, never to choose it.
PROBE_LATS = (58.3, -56.2, 86.0)
_R_KM = 6371.0088


def _area_uniform_lat(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Latitude in [lo, hi] (degrees) with density proportional to cos(lat), i.e. uniform over the sphere's area."""
    s = rng.uniform(math.sin(math.radians(lo)), math.sin(math.radians(hi)))
    return math.degrees(math.asin(s))


def _gc_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return _R_KM * math.acos(max(-1.0, min(1.0, c)))


def probe_distance(lat: float, lon: float, low_lon: float) -> dict:
    probes = [(58.3, 6.0), (-56.2, 12.0), (86.0, low_lon)]
    return {
        "probe_lat_dist": round(min(abs(lat - p) for p in PROBE_LATS), 3),
        "probe_km": round(min(_gc_km(lat, lon, plat, plon) for plat, plon in probes), 1),
    }


def _raster_half_span_deg(spec, lat: float) -> float:
    ext = max((r.extent_m for r in spec.rasters.values()), default=0.0)
    return math.degrees(1.6 * ext / (_R_KM * 1000.0 * max(math.cos(math.radians(lat)), 1e-6)))


def sample_local(spec, rng: np.random.Generator) -> list[tuple[float, float, str]]:
    out = []
    for lo, hi in BANDS:
        for _ in range(PER_BAND):
            lat = _area_uniform_lat(rng, lo, hi) * float(rng.choice([-1.0, 1.0]))
            lon = float(rng.uniform(-180.0, 180.0))
            if spec.rasters:  # generator limitation: GeoTIFF scenes cannot straddle 180 deg
                lim = 180.0 - _raster_half_span_deg(spec, lat) - 0.01
                while abs(lon) > lim:
                    lon = float(rng.uniform(-180.0, 180.0))
            out.append((lat, lon, f"{int(lo)}-{int(hi)}"))
    return out


def sample_large(spec, rng: np.random.Generator, k: int) -> list[tuple[float, float, str]]:
    b = np.array([g.total_bounds for g in spec.layers.values()])
    minx, miny, maxx, maxy = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    lat_lo, lat_hi = -80.0 - miny, 80.0 - maxy
    lon_lo, lon_hi = -179.9 - minx, 179.9 - maxx
    assert lat_lo < lat_hi and lon_lo < lon_hi, (minx, miny, maxx, maxy)
    return [(_area_uniform_lat(rng, lat_lo, lat_hi), float(rng.uniform(lon_lo, lon_hi)), "large") for _ in range(k)]


def _crs_name(c) -> str:
    c = CRS.from_user_input(c)
    auth = c.to_authority()
    return ":".join(auth) if auth else c.name


def raster_regime(lat: float) -> str:
    a = abs(lat)
    return "low" if a <= 2.2 else ("high" if a <= 85.0 else "polar")


def write_blind_variant(spec, files: dict[str, str], lat: float, lon: float, out: Path, *,
                        store_proj_gpkg: bool, large: bool) -> dict:
    """Write one relocated workspace; return what was stored where (mirrors build.write_variant)."""
    out.mkdir(parents=True, exist_ok=True)
    storage: dict[str, str] = {}
    crosses = False
    for stem, gdf in spec.layers.items():
        geoms = ([shapely.affinity.translate(x, lon, lat) for x in gdf.geometry] if large
                 else place(list(gdf.geometry), lat, lon))
        g = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=geoms, crs="EPSG:4326")
        bx = g.total_bounds
        crosses |= bool(bx[2] - bx[0] > 180.0)
        fname = files[stem]
        if fname.endswith(".csv"):
            df = pd.DataFrame(g.drop(columns="geometry"))
            df["lon"], df["lat"] = g.geometry.x.round(7), g.geometry.y.round(7)
            df.to_csv(out / fname, index=False)
            storage[stem] = "EPSG:4326 (csv)"
        elif fname.endswith(".gpkg") and store_proj_gpkg and not large and not (bx[2] - bx[0] > 180.0):
            dst = ("EPSG:3413" if lat > 84.0 else "EPSG:3031" if lat < -80.0 else g.estimate_utm_crs())
            g.to_crs(dst).to_file(out / fname, driver="GPKG")
            storage[stem] = _crs_name(dst)
        else:
            g.to_file(out / fname, driver="GPKG" if fname.endswith(".gpkg") else "GeoJSON")
            storage[stem] = "EPSG:4326"
    for stem, rspec in spec.rasters.items():
        dst_crs, nodata = storage_crs_for_raster(raster_regime(lat), lat, force_4326_high=rspec.force_4326_high)
        write_physical_geotiff(out / files[stem], lat=lat, lon=lon, extent_m=rspec.extent_m, pixel_m=rspec.pixel_m,
                               field_fn=rspec.field_fn, valid_fn=rspec.valid_fn, dst_crs=dst_crs, nodata=nodata)
        storage[stem] = f"{_crs_name(dst_crs)} nodata={nodata:g}"
    return {"storage": storage, "crosses_antimeridian": crosses}


def solve_workspace(fam, spec, files: dict[str, str], d: Path) -> float:
    layers, rasters = B.read_workspace(d, files)
    if fam.large:
        return float(fam.solve(layers, rasters, None, None, spec.params))
    crs_obj = reference_aeqd_crs(layers, rasters)
    return float(fam.solve(layers, rasters, lambda g, crs=crs_obj: g.to_crs(crs), crs_obj, spec.params))


def _same(a: dict, b: dict) -> bool:
    if a["files"] != b["files"] or a["store_proj_gpkg"] != b["store_proj_gpkg"] or set(a["variants"]) != set(b["variants"]):
        return False
    for v, x in b["variants"].items():
        y = a["variants"][v]
        if (y["lat"], y["lon"]) != (x["lat"], x["lon"]) or abs(y["truth"] - x["truth"]) > 1e-9 * max(1.0, abs(x["truth"])):
            return False
    return True


def _family_worker(args: tuple) -> list[dict]:
    fam_idx, v2_root, out_root, v2_seed, blind_seed, n_train, n_test, n_heldout = args
    fam = FAMILIES[fam_idx]
    stored = {t["task_id"]: t for t in map(json.loads, open(Path(v2_root) / "tasks.jsonl")) if t["family"] == fam.name}
    captured: dict = {}
    orig_build = fam.build

    def capturing_build(rng):
        captured["spec"] = orig_build(rng)
        return captured["spec"]

    fam.build = capturing_build
    fam_rng = np.random.default_rng(np.random.SeedSequence([v2_seed, fam_idx]))
    plan = [("test", n_heldout)] if fam.held_out else [("train", n_train), ("test", n_test)]
    tmp = Path(tempfile.mkdtemp(prefix=f"blind_{fam.name}_"))
    out_tasks = []
    try:
        for split, n in plan:
            for i in range(n):
                tid = f"{fam.name}-{split}-{i:04d}"
                t = B.make_task(fam, fam_rng, tmp, tid, split)
                shutil.rmtree(tmp / tid, ignore_errors=True)
                if not _same(t, stored[tid]):
                    raise RuntimeError(f"replay of {tid} does not match data/v2 (library versions changed?)")
                if split != "test":
                    continue
                spec, low = captured["spec"], t["variants"]["low"]
                rng = np.random.default_rng(np.random.SeedSequence([blind_seed, fam_idx, i]))
                centres = sample_large(spec, rng, len(BANDS) * PER_BAND) if fam.large else sample_local(spec, rng)
                variants = {}
                for k, (lat, lon, band) in enumerate(centres):
                    vn = f"b{k}"
                    d = Path(out_root) / tid / vn
                    if d.exists():
                        shutil.rmtree(d)
                    info = write_blind_variant(spec, t["files"], lat, lon, d,
                                               store_proj_gpkg=t["store_proj_gpkg"], large=fam.large)
                    truth = solve_workspace(fam, spec, t["files"], d)
                    rel = abs(truth - low["truth"]) / max(abs(low["truth"]), 1e-12)
                    v = {"dir": f"{tid}/{vn}", "lat": round(lat, 5), "lon": round(lon, 5), "band": band,
                         "truth": truth, "finite": bool(np.isfinite(truth)), **info,
                         **probe_distance(lat, lon, low["lon"])}
                    if not fam.large:  # same content, so the answer must not move: build.py's criterion
                        v["rel_diff_vs_low"] = rel
                        v["invariant"] = bool(np.isfinite(truth) and rel <= (0.0 if t["answer"] == "count" else 0.008))
                    # Not scored: ill-posed local variants (storage resampling moves the reference answer by about
                    # the scoring tolerance, so a correct script could legitimately miss it)
                    v["scored"] = bool(v["finite"] and v.get("invariant", True))
                    variants[vn] = v
                out_tasks.append({
                    "task_id": tid, "family": fam.name, "answer": t["answer"], "large": fam.large,
                    "has_raster": fam.has_raster, "held_out": fam.held_out, "files": t["files"],
                    "store_proj_gpkg": t["store_proj_gpkg"],
                    "low": {"lat": low["lat"], "lon": low["lon"], "truth": low["truth"]},
                    "variants": variants,
                })
    finally:
        fam.build = orig_build
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {fam.name:28s} {len(out_tasks)} test tasks x {len(BANDS) * PER_BAND} blind variants", flush=True)
    return out_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v2", default="data/v2", help="existing GSB v2 root (tasks.jsonl is replayed and checked)")
    ap.add_argument("--out", default="data/v2_blind")
    ap.add_argument("--v2-seed", type=int, default=0)
    ap.add_argument("--blind-seed", type=int, default=20261125)
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--heldout", type=int, default=30)
    ap.add_argument("--families", default="", help="comma-separated subset (smoke runs)")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pick = set(filter(None, a.families.split(",")))
    jobs = [(i, a.v2, str(out), a.v2_seed, a.blind_seed, a.train, a.test, a.heldout)
            for i, f in enumerate(FAMILIES) if not pick or f.name in pick]
    tasks: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
        for ft in ex.map(_family_worker, jobs):
            tasks.extend(ft)
    with open(out / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    vs = [v for t in tasks for v in t["variants"].values()]
    local = [v for t in tasks if not t["large"] for v in t["variants"].values()]
    meta = {
        "blind_seed": a.blind_seed, "v2_seed": a.v2_seed, "bands": BANDS, "per_band": PER_BAND,
        "n_tasks": len(tasks), "n_variants": len(vs),
        "non_finite_truths": sum(not v["finite"] for v in vs),
        "local_not_invariant": sum(not v["invariant"] for v in local),
        "scored": sum(v["scored"] for v in vs),
        "not_scored_by_family_band": dict(Counter(f'{t["family"]}|{v["band"]}' for t in tasks
                                                  for v in t["variants"].values() if not v["scored"])),
        "crosses_antimeridian": sum(v["crosses_antimeridian"] for v in vs),
        "by_band": dict(Counter(v["band"] for v in vs)),
    }
    (out / "blind_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

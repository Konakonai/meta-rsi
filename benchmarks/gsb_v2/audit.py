"""Method-separability audit for GSB v2.

Verifies across every generated task:
  - reference  : local circular-mean AEQD (or WGS84 Geod for large) -> must pass 100%
  - utm        : naive estimate_utm_crs() -> fails polar (100%), antimeridian, and large-extent
  - webmercator: EPSG:3857 -> spuriously passes `low` (|lat|<=2.2 deg) while failing high/polar/antimeridian
  - low_only   : single-location `low` accuracy of EPSG:3857 (proving true weak-proxy unidentifiability)
  - naive_nodata: (for raster families) reading raster in native CRS ignoring nodata sentinel

Usage:
    python -m benchmarks.gsb_v2.audit data/v2
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS

from benchmarks.gsb.harness import matches

from .build import read_workspace
from .families import BY_NAME
from .geo import reference_aeqd_crs


def _naive_raster_eval(fam_name: str, layers, rasters, params) -> float | None:
    """Evaluate raster family naively in its native storage CRS while ignoring nodata (or using EPSG:3857 directly)."""
    rpath = next(iter(rasters.values()))
    with rasterio.open(rpath) as src:
        arr = src.read(1).astype(np.float64)
        tr = src.transform
        rcrs = src.crs
    dx, dy = abs(tr.a), abs(tr.e)
    pix_area = dx * dy

    def _proj(g):
        return g.to_crs(rcrs)

    if fam_name == "raster_slope_exceed_area":
        import rasterio.features
        import shapely

        b = _proj(layers["basin"])
        u = shapely.union_all(list(b.geometry))
        m = rasterio.features.geometry_mask([u], out_shape=arr.shape, transform=tr, invert=True)
        dz_dy, dz_dx = np.gradient(arr, dy, dx)
        slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
        return float(np.sum(m & (slope > params["slope_deg"])) * pix_area * 1e-4)

    # For vector+raster families, project vector to raster CRS and ignore nodata sentinel
    import rasterio.features
    import shapely

    if fam_name == "raster_zonal_mean":
        z = _proj(layers["zones"])
        u = shapely.union_all(list(z[z["landuse"] == params["landuse"]].geometry))
        m = rasterio.features.geometry_mask([u], out_shape=arr.shape, transform=tr, invert=True)
        vals = arr[m]
        return float(np.mean(vals)) if len(vals) else 0.0
    if fam_name == "raster_threshold_area":
        z = _proj(layers["zones"])
        u = shapely.union_all(list(z[z["landuse"] == params["landuse"]].geometry))
        m = rasterio.features.geometry_mask([u], out_shape=arr.shape, transform=tr, invert=True)
        unit_scale = 1e-6 if params["unit"] == "km2" else 1e-4
        return float(np.sum(m & (arr >= params["threshold"])) * pix_area * unit_scale)
    if fam_name == "raster_buffer_extract":
        s = _proj(layers["sensors"])
        u = shapely.union_all(list(s[s["status"] == "active"].geometry.buffer(params["radius"])))
        m = rasterio.features.geometry_mask([u], out_shape=arr.shape, transform=tr, invert=True)
        vals = arr[m]
        return float(np.mean(vals)) if len(vals) else 0.0
    if fam_name == "raster_pop_exposure":
        b = _proj(layers["blocks"])
        tot = 0.0
        for geom, pop in zip(b.geometry, b["population"]):
            m = rasterio.features.geometry_mask([geom], out_shape=arr.shape, transform=tr, invert=True)
            if np.any(m) and float(np.mean(arr[m])) >= params["threshold"]:
                tot += float(pop)
        return tot
    if fam_name == "raster_corridor_elevation":
        r = _proj(layers["roads"])
        u = shapely.union_all(list(r[r["road_type"] == "primary"].geometry.buffer(params["width"])))
        m = rasterio.features.geometry_mask([u], out_shape=arr.shape, transform=tr, invert=True)
        vals = arr[m]
        return float(np.mean(vals)) if len(vals) else 0.0
    if fam_name == "raster_hotspot_in_buffer":
        f = _proj(layers["facilities"])
        u = shapely.union_all(list(f.geometry.buffer(params["radius"])))
        m = rasterio.features.geometry_mask([u], out_shape=arr.shape, transform=tr, invert=True)
        return float(np.sum(m & (arr >= params["threshold"])) * pix_area * 1e-6)
    return None


def _eval_method(fam, layers, rasters, params, mode: str) -> float:
    if mode == "reference":
        if fam.large:
            return float(fam.solve(layers, rasters, None, None, params))
        crs = reference_aeqd_crs(layers, rasters)
        return float(fam.solve(layers, rasters, lambda g: g.to_crs(crs), crs, params))

    if mode == "utm":
        if layers:
            crs = next(iter(layers.values())).to_crs("EPSG:4326").estimate_utm_crs()
        else:
            # Raster-only: estimate UTM from raster center
            rpath = next(iter(rasters.values()))
            with rasterio.open(rpath) as src:
                from pyproj import Transformer
                import geopandas as gpd
                from shapely.geometry import Point
                cx = 0.5 * (src.bounds.left + src.bounds.right)
                cy = 0.5 * (src.bounds.bottom + src.bounds.top)
                lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
                crs = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").estimate_utm_crs()
        return float(fam.solve(layers, rasters, lambda g: g.to_crs(crs), crs, params))

    if mode == "webmercator":
        crs = CRS.from_epsg(3857)
        if fam.has_raster:
            # Check bounds validity in EPSG:3857 (undefined above 85.05N)
            rpath = next(iter(rasters.values()))
            with rasterio.open(rpath) as src:
                from pyproj import Transformer
                cx = 0.5 * (src.bounds.left + src.bounds.right)
                cy = 0.5 * (src.bounds.bottom + src.bounds.top)
                _, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
                if abs(lat) > 85.0:
                    raise ValueError("Out of EPSG:3857 latitude domain")
            # For raster tasks, evaluate directly in EPSG:3857 (keeping nodata valid to isolate projection error!)
            return float(fam.solve(layers, rasters, lambda g: g.to_crs(crs), crs, params))
        def _p(g):
            r = g.to_crs(crs)
            if not np.isfinite(r.total_bounds).all():
                raise ValueError("Non-finite EPSG:3857")
            return r
        return float(fam.solve(layers, rasters, _p, crs, params))

    if mode == "naive_raster":
        v = _naive_raster_eval(fam.name, layers, rasters, params)
        if v is None:
            raise ValueError("not raster")
        return v

    raise ValueError(mode)


def _audit_one_family(args: tuple[str, str, list[dict]]) -> tuple[str, dict[str, int]]:
    fam_name, root_str, fam_tasks = args
    root = Path(root_str)
    fam = BY_NAME[fam_name]
    f: dict[str, int] = defaultdict(
        int,
        {
            "n": 0,
            "reference_pass": 0,
            "utm_pass": 0,
            "webmercator_pass": 0,
            "wm_low_pass": 0,
            "webmercator_lucky": 0,
            "naive_nodata_caught": 0,
        },
    )
    for t in fam_tasks:
        f["n"] += 1
        for mode in ("reference", "utm", "webmercator"):
            oks = []
            for v, info in t["variants"].items():
                layers, rasters = read_workspace(root / info["dir"], t["files"])
                try:
                    val = _eval_method(fam, layers, rasters, t["params"], mode)
                    ok = bool(np.isfinite(val) and matches(val, info["truth"], t["answer"]))
                except Exception:
                    ok = False
                oks.append(ok)
                if mode == "webmercator" and v == "low":
                    f["wm_low_pass"] += int(ok)
            if all(oks):
                f[f"{mode}_pass"] += 1
            elif any(oks):
                f[f"{mode}_lucky"] += 1
            else:
                f[f"{mode}_fail"] += 1
        if fam.has_raster:
            noks = []
            for v, info in t["variants"].items():
                layers, rasters = read_workspace(root / info["dir"], t["files"])
                try:
                    val = _eval_method(fam, layers, rasters, t["params"], "naive_raster")
                    ok = bool(np.isfinite(val) and matches(val, info["truth"], t["answer"]))
                except Exception:
                    ok = False
                noks.append(ok)
            f["naive_nodata_caught"] += int(not all(noks))
    return fam_name, dict(f)


def audit_dataset(root: Path, limit_per_family: int = 0, workers: int = 10) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    tasks = [json.loads(ln) for ln in open(root / "tasks.jsonl")]
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        fam_name = t["family"]
        if limit_per_family and len(by_fam[fam_name]) >= limit_per_family:
            continue
        by_fam[fam_name].append(t)

    jobs = [(fn, str(root), by_fam[fn]) for fn in sorted(by_fam)]
    table: dict[str, dict[str, int]] = {}
    if workers <= 1 or len(tasks) <= 40:
        for j in jobs:
            fn, res = _audit_one_family(j)
            table[fn] = res
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            for fn, res in ex.map(_audit_one_family, jobs):
                table[fn] = res

    cols = ["n", "reference_pass", "utm_pass", "webmercator_pass", "wm_low_pass", "webmercator_lucky", "naive_nodata_caught"]
    print("family".ljust(28) + "".join(c[:11].rjust(12) for c in cols))
    print("-" * (28 + 12 * len(cols)))
    for fam_name in sorted(table):
        f = defaultdict(int, table[fam_name])
        print(fam_name.ljust(28) + "".join(str(f[c]).rjust(12) for c in cols))
    return table


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/v2")
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    audit_dataset(root, limit_per_family=lim)


if __name__ == "__main__":
    main()

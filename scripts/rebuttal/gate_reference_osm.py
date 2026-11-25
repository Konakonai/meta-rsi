"""Does the metamorphic gate reject CORRECT solutions on real OSM tasks, invariant and non-invariant alike?

Runs a hand-written reference solution for every data/osm_global and data/osm_eval task (12 templates; every
reference must reproduce the stored truth), then puts each reference through the gate used for `base_meta`,
`sft_meta` and `meta_gate_only`: the built-in probes plus the 3 certified probes. Each probe is run and scored
separately (no early exit), so every rejection is attributed. The two `abs_pos` templates have answers that
legitimately change when the data is relocated; a gate that relocates data must reject their correct solutions.

    nix-shell --run "python scripts/rebuttal/gate_reference_osm.py"   # -> runs/v11_gate_precision/
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.gsb.harness import matches  # noqa: E402
from meta_rsi.evaluator import _run_metamorphic_check  # noqa: E402
from meta_rsi.metamorphic import perturb_workspace  # noqa: E402
from meta_rsi.sandbox import parse_final_number, run_python_sandbox  # noqa: E402
from meta_rsi.training.collect_trajectories import load_probes_from_dir  # noqa: E402

OUT = REPO / "runs/v11_gate_precision"

PRE = '''
import math
import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.ops import unary_union

def load(name):
    g = gpd.read_file(name)
    return g.set_crs("EPSG:4326") if g.crs is None else g

def local_crs(*gdfs):
    """Azimuthal equidistant projection centred on the data, whatever CRS it is stored in."""
    b = np.array([g.to_crs("EPSG:4326").total_bounds for g in gdfs])
    lons = np.radians([b[:, 0].min(), b[:, 2].max()])
    lon0 = math.degrees(math.atan2(np.sin(lons).mean(), np.cos(lons).mean()))
    lat0 = (b[:, 1].min() + b[:, 3].max()) / 2
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
'''

REF = {
    "osm_building_area_ha": "b = load('buildings.gpkg')\nprint(b.to_crs(local_crs(b)).area.sum() / 1e4)",
    "osm_road_length_km": "r = load('roads.gpkg')\nprint(r.to_crs(local_crs(r)).length.sum() / 1e3)",
    "osm_poi_buffer_building_area": (
        "b, p = load('buildings.gpkg'), load('pois.gpkg')\nc = local_crs(b, p)\n"
        "u = unary_union(p.to_crs(c).buffer(150.0))\nprint(b.to_crs(c).intersection(u).area.sum() / 1e4)"),
    "osm_large_building_area_ha": "b = load('buildings.gpkg')\na = b.to_crs(local_crs(b)).area\nprint(a[a > 300.0].sum() / 1e4)",
    "osm_nearest_poi_median_dist": (
        "b, p = load('buildings.gpkg'), load('pois.gpkg')\nc = local_crs(b, p)\nu = unary_union(p.to_crs(c).geometry)\n"
        "print(float(np.median([pt.distance(u) for pt in b.to_crs(c).centroid])))"),
    "osm_road_buffer_poi_count": (
        "r, p = load('roads.gpkg'), load('pois.gpkg')\nc = local_crs(r, p)\n"
        "u = unary_union(r.to_crs(c).buffer(30.0))\nprint(int(p.to_crs(c).within(u).sum()))"),
    "osm_building_perimeter_km": "b = load('buildings.gpkg')\nprint(b.to_crs(local_crs(b)).length.sum() / 1e3)",
    "osm_poi_nn_spacing_m": (
        "p = load('pois.gpkg')\nxy = np.array([(q.x, q.y) for q in p.to_crs(local_crs(p)).geometry])\n"
        "d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])\nnp.fill_diagonal(d, np.inf)\n"
        "print(float(np.median(d.min(axis=1))))"),
    "osm_road_hull_density": (
        "r = load('roads.gpkg')\nrr = r.to_crs(local_crs(r))\n"
        "print((rr.length.sum() / 1e3) / (unary_union(rr.geometry).convex_hull.area / 1e6))"),
    # Topological control: the stored truth tests intersection on the WGS84 geometries.
    "osm_bldg_road_intersects_cnt": (
        "b, r = load('buildings.gpkg').to_crs(4326), load('roads.gpkg').to_crs(4326)\n"
        "print(int(b.intersects(unary_union(r.geometry)).sum()))"),
    "osm_northernmost_bldg_lat": (
        "b = load('buildings.gpkg')\nc = local_crs(b)\nlat = gpd.GeoSeries(b.to_crs(c).centroid, crs=c).to_crs(4326).y\n"
        "print(repr(float(lat.max())))"),
    "osm_north_half_bldg_count": (
        "b = load('buildings.gpkg')\nc = local_crs(b)\n_, miny, _, maxy = b.to_crs(4326).total_bounds\n"
        "lat = gpd.GeoSeries(b.to_crs(c).centroid, crs=c).to_crs(4326).y\nprint(int((lat > (miny + maxy) / 2).sum()))"),
}


def per_probe(code: str, src: Path, orig: float, answer: str, probes, timeout: float = 25.0) -> dict:
    """Every gate check on its own, mirroring _run_metamorphic_check's checks and scoring without early exit."""
    checks = [("shift_and_shuffle", dict(target_lat=58.3, target_lon=6.0, shuffle_rows=True, flip_gpkg_crs=False, seed=42))]
    if any(p.suffix.lower() in {".gpkg", ".tif", ".tiff"} or p.name == "metadata.json" for p in src.iterdir() if p.is_file()):
        checks.append(("crs_storage_flip", dict(target_lat=-56.2, target_lon=12.0, shuffle_rows=True, flip_gpkg_crs=True, seed=99)))
    checks += list(probes)
    res = {}
    for name, spec in checks:
        with tempfile.TemporaryDirectory() as tmp:
            pert = Path(tmp) / "pert"
            if callable(spec):
                pert.mkdir(parents=True)
                spec(src, pert)
            else:
                perturb_workspace(src, pert, **spec)
            r = run_python_sandbox(code, pert, timeout=timeout)
        if r["timed_out"] or r["returncode"] != 0:
            res[name] = {"status": "timeout" if r["timed_out"] else "crash",
                         "stderr": (r["stderr"].strip().splitlines() or [""])[-1][:200]}
            continue
        v = parse_final_number(r["stdout"])
        res[name] = {"status": "no_number"} if v is None else {"status": "ok", "value": v, "consistent": matches(v, orig, answer)}
    return res


def main() -> None:
    probes = load_probes_from_dir(REPO / "runs/v8_s4_wave2_v2/certified_probes", OUT / "_probe_mods")
    assert probes and len(probes) == 3
    jobs = []
    for root in ("data/osm_global", "data/osm_eval"):
        for t in map(json.loads, open(REPO / root / "tasks.jsonl")):
            jobs.append((root, t))

    def one(job):
        root, t = job
        bf = t.get("base_family") or t["family"].split("__")[0]
        code = PRE + REF[bf] + "\n"
        low = REPO / root / t["variants"]["low"]["dir"]
        truth = t["variants"]["low"]["truth"]
        r = run_python_sandbox(code, low, timeout=60)
        val = parse_final_number(r["stdout"]) if r["returncode"] == 0 else None
        rec = {"set": root.split("/")[-1], "task_id": t["task_id"], "template": bf, "answer": t["answer"],
               "truth": truth, "value": val, "ref_correct": val is not None and matches(val, truth, t["answer"])}
        if not rec["ref_correct"]:
            rec["stderr"] = r["stderr"][-300:]
            return rec
        ok, diag, _ = _run_metamorphic_check(code, low, val, t["answer"], custom_probes=probes)
        rec["gate_pass"] = ok
        rec["probes"] = per_probe(code, low, val, t["answer"], probes)
        rec["probes_agree"] = ok == all(p.get("consistent") for p in rec["probes"].values())
        return rec

    with ThreadPoolExecutor(max_workers=16) as ex:
        recs = list(ex.map(one, jobs))
    with open(OUT / "osm_reference_gate.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    bad = [r for r in recs if not r["ref_correct"]]
    print(f"reference solutions correct: {len(recs) - len(bad)}/{len(recs)}")
    for r in bad:
        print("  WRONG", r["task_id"], r["truth"], r["value"], r.get("stderr", "")[-120:])
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r["ref_correct"]:
            by_t[r["template"]].append(r)
    summary = {}
    print(f"\n{'template':32s} {'n':>3s} {'gate pass':>9s}  rejecting probes")
    for tpl, rs in sorted(by_t.items(), key=lambda kv: (kv[0].startswith("osm_north"), kv[0])):
        rej = Counter(name for r in rs for name, p in r["probes"].items() if not p.get("consistent"))
        n_pass = sum(r["gate_pass"] for r in rs)
        summary[tpl] = {"n": len(rs), "gate_pass": n_pass, "rejections_by_probe": dict(rej)}
        print(f"{tpl:32s} {len(rs):3d} {n_pass:5d}/{len(rs):<3d}  {dict(rej)}")
    summary["_probe_mirror_agrees_with_gate"] = all(r.get("probes_agree", True) for r in recs)
    (OUT / "osm_reference_gate_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nper-probe mirror agrees with _run_metamorphic_check:", summary["_probe_mirror_agrees_with_gate"])


if __name__ == "__main__":
    main()

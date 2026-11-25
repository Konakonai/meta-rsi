"""Why does the metamorphic gate reject scripts the oracle calls correct?

Takes every false rejection found by gate_precision.py (gate fails, oracle correct), rebuilds the workspace of the
probe that rejected it, reruns the script there and assigns the first matching cause:
  probe_changes_truth  the family's reference solver, run on the probe's workspace, no longer gives the `low` truth:
                       the probe is not answer-preserving for this task and the gate is wrong (all large_* tasks,
                       plus thresholded aggregates that flip when a raster is resampled)
  mixed_crs_unhandled  the probe leaves layers in different CRSs (a GeoPackage flipped to UTM next to a WGS84
                       GeoJSON, or two layers on either side of a UTM zone edge) and the script combines them without
                       reprojecting (GeoPandas CRS-mismatch warning): a latent bug this task's oracle variants never
                       exercise, because they store every layer in one CRS
  km_unit_bug          rejected by the km-unit probe, and consistent again under the same probe with `+units=m`: the
                       script reads projected coordinates as metres; a latent bug, no oracle variant stores data in km
  projected_storage    the reference answer survives a storage-only probe (Web Mercator, or the transverse Mercator
                       km probe even with metre units), the script's does not: it treats stored coordinates as WGS84
                       degrees (e.g. pyproj.Geod on raw x/y -> NaN) or measures in the stored projection
  polar_wrap           the polar probe moves a continental-size scene over the pole so that it spans more than 180
                       degrees of longitude; the reference (geodesic ring area) still holds, planar methods do not.
                       The oracle's large-extent variants stay within +-140 degrees, so such inputs are out of its scope
  other                anything else (listed with the script's output for reading)

    nix-shell --run "python scripts/rebuttal/gate_rejection_causes.py"   # -> runs/v11_gate_precision/
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.gsb.harness import matches  # noqa: E402
from benchmarks.gsb_v2.build import read_workspace  # noqa: E402
from benchmarks.gsb_v2.families import BY_NAME  # noqa: E402
from benchmarks.gsb_v2.geo import reference_aeqd_crs  # noqa: E402
from meta_rsi.metamorphic import perturb_workspace  # noqa: E402
from meta_rsi.verifier_agent import load_probe_fn  # noqa: E402
from meta_rsi.sandbox import parse_final_number, run_python_sandbox  # noqa: E402
from meta_rsi.training.collect_trajectories import load_probes_from_dir  # noqa: E402

OUT = REPO / "runs/v11_gate_precision"
DATA = REPO / "data/v2"
POPS = ("base_weak", "same_teacher_weak", "darwinx_weak")
BUILTIN = {
    "shift_and_shuffle": dict(target_lat=58.3, target_lon=6.0, shuffle_rows=True, flip_gpkg_crs=False, seed=42),
    "crs_storage_flip": dict(target_lat=-56.2, target_lon=12.0, shuffle_rows=True, flip_gpkg_crs=True, seed=99),
}
_CRS_WARN = ("CRS mismatch", "Use `to_crs()` to reproject one of the input geometries")


def _crs_of(f: Path) -> str | None:
    if f.suffix == ".csv":
        return "EPSG:4326"
    if f.suffix in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(f) as r:
            return r.crs.to_string() if r.crs else None
    if f.suffix in (".gpkg", ".geojson", ".shp"):
        import geopandas as gpd
        c = gpd.read_file(f, rows=1).crs
        return c.to_string() if c else None
    return None


def _crs_set(d: Path) -> list[str]:
    return sorted({c for f in d.iterdir() if f.is_file() and (c := _crs_of(f))})


def _code(row: dict) -> str:
    return re.search(r"```(?:python)?\s*\n(.*?)```", row["assistant"], re.S).group(1)


def _lon_span(d: Path) -> float:
    import geopandas as gpd
    spans = [0.0]
    for f in d.iterdir():
        if f.suffix in (".gpkg", ".geojson"):
            g = gpd.read_file(f)
            if g.crs is not None and len(g):
                b = g.to_crs(4326).total_bounds
                spans.append(float(b[2] - b[0]))
    return max(spans)


def reference_truth(task: dict, d: Path) -> float:
    fam = BY_NAME[task["family"]]
    layers, rasters = read_workspace(d, task["files"])
    if fam.large:
        return float(fam.solve(layers, rasters, None, None, task["params"]))
    crs = reference_aeqd_crs(layers, rasters)
    return float(fam.solve(layers, rasters, lambda g: g.to_crs(crs), crs, task["params"]))


def _perturb(name: str, src: Path, dst: Path, probes: dict) -> None:
    if name in BUILTIN:
        perturb_workspace(src, dst, **BUILTIN[name])
    else:
        dst.mkdir(parents=True)
        probes[name](src, dst)


def classify(rec: dict, code: str, task: dict, probes: dict) -> dict:
    src = DATA / task["variants"]["low"]["dir"]
    orig, truth = rec["variants"]["low"]["value"], task["variants"]["low"]["truth"]
    name = rec["fail_probe"]
    with tempfile.TemporaryDirectory() as tmp:
        pert = Path(tmp) / "pert"
        _perturb(name, src, pert, probes)
        crs_pert = _crs_set(pert)
        lon_span = _lon_span(pert)
        try:
            truth_pert = reference_truth(task, pert)
        except Exception as e:  # noqa: BLE001
            truth_pert = f"solver failed: {type(e).__name__}: {e}"[:200]
        r = run_python_sandbox(code, pert, timeout=25.0)
        control = None
        if name == "probe_r3_km_units":  # the same probe with metre units: does the inconsistency go away?
            ctl = Path(tmp) / "ctl"
            _perturb("probe_r3_m_units_control", src, ctl, probes)
            rc = run_python_sandbox(code, ctl, timeout=25.0)
            control = parse_final_number(rc["stdout"]) if rc["returncode"] == 0 and not rc["timed_out"] else None
    val = parse_final_number(r["stdout"]) if r["returncode"] == 0 and not r["timed_out"] else None
    out = {"task_id": rec["task_id"], "family": rec["family"], "probe": name, "kind": rec["fail_kind"],
           "orig_value": orig, "probe_value": val, "truth_low": truth, "truth_probe": truth_pert,
           "m_units_control_value": control, "crs_low": _crs_set(src), "crs_probe": crs_pert,
           "probe_lon_span": round(lon_span, 1), "crs_warning": any(w in r["stderr"] for w in _CRS_WARN)}
    if isinstance(truth_pert, float) and not matches(truth_pert, truth, task["answer"]):
        out["cause"] = "probe_changes_truth"
    elif out["crs_warning"]:
        out["cause"] = "mixed_crs_unhandled"
    elif control is not None and matches(control, orig, task["answer"]):
        out["cause"] = "km_unit_bug"
    elif name in ("probe_r1_aligned_wm", "probe_r3_km_units"):
        out["cause"] = "projected_storage"
    elif name == "probe_r2_polar_laea" and lon_span > 180.0:
        out["cause"] = "polar_wrap"
    else:
        out["cause"] = "other"
        out["stdout_tail"] = r["stdout"][-200:]
        out["stderr_tail"] = (r["stderr"].strip().splitlines() or [""])[-1][:200]
    return out


def main() -> None:
    tasks = {t["task_id"]: t for t in map(json.loads, open(DATA / "tasks.jsonl"))}
    pdir = REPO / "runs/v8_s4_wave2_v2/certified_probes"
    probes = dict(load_probes_from_dir(pdir, OUT / "_probe_mods"))
    km_src = (pdir / "probe_r3_km_units.py").read_text()
    assert km_src.count("+units=km") == 1
    probes["probe_r3_m_units_control"] = load_probe_fn(km_src.replace("+units=km", "+units=m"), OUT / "_probe_mods",
                                                       name="probe_r3_m_units_control")
    jobs = []
    for pop in POPS:
        rows = [json.loads(ln) for ln in open(OUT / "trajectories" / f"{pop}.jsonl")]
        recs = [json.loads(ln) for ln in open(OUT / f"gate_vs_oracle_{pop}.jsonl")]
        assert len(rows) == len(recs)  # gate_precision.py writes one record per script, in trajectory order
        for row, rec in zip(rows, recs):
            assert row["task_id"] == rec["task_id"]
            if rec["gate"] == "fail" and rec["oracle"] == "correct":
                jobs.append((pop, rec, _code(row)))
    with ThreadPoolExecutor(max_workers=18) as ex:
        res = list(ex.map(lambda j: {"population": j[0]} | classify(j[1], j[2], tasks[j[1]["task_id"]], probes), jobs))
    with open(OUT / "false_rejection_causes.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    summary: dict = {}
    for pop in POPS:
        rs = [r for r in res if r["population"] == pop]
        summary[pop] = {"n": len(rs), "by_cause": dict(Counter(r["cause"] for r in rs).most_common()),
                        "by_cause_probe": dict(Counter(f"{r['cause']}|{r['probe']}" for r in rs).most_common())}
    (OUT / "false_rejection_causes_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    by = defaultdict(list)
    for r in res:
        if r["cause"] == "other":
            by[r["population"]].append(r)
    for pop, rs in by.items():
        print(f"\n[other] {pop}")
        for r in rs:
            print(f"  {r['task_id']:40s} {r['probe']:22s} {r['kind']:12s} orig={r['orig_value']:.6g} probe={r['probe_value']} "
                  f"ctl={r['m_units_control_value']} truth={r['truth_low']:.6g}->{r['truth_probe']} "
                  f"crs_probe={[c[:24] for c in r['crs_probe']]} | {r.get('stderr_tail', '')[:80]}")


if __name__ == "__main__":
    main()

"""Comprehensive unit tests for GSB v2, GeoTIFF metamorphic perturbations, and leak-free multi-modal verifier certification."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.gsb.harness import matches
from benchmarks.gsb_v2.audit import audit_dataset
from benchmarks.gsb_v2.build import make_task, read_workspace
from benchmarks.gsb_v2.families import BY_NAME, FAMILIES
from benchmarks.gsb_v2.geo import reference_aeqd_crs
from meta_rsi.metamorphic import perturb_workspace
from meta_rsi.verifier_agent import VERIFIER_SYSTEM_PROMPT, certify_perturbation_operator


def test_verifier_system_prompt_has_no_failure_mode_hints():
    """Ensure Verifier-Proposer prompt contains zero explicit geographic/raster failure-mode hints."""
    low = VERIFIER_SYSTEM_PROMPT.lower()
    for forbidden in ("84n", "80s", "utm", "antimeridian", "3413", "3031", "nodata", "3857"):
        assert forbidden not in low, f"Verifier prompt leaked failure hint: {forbidden!r}"


def test_gsb_v2_all_20_families_build_and_audit_separability():
    assert len(FAMILIES) == 20
    assert sum(1 for f in FAMILIES if not f.held_out) == 15
    assert sum(1 for f in FAMILIES if f.held_out) == 5

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tasks = []
        for fam in FAMILIES:
            t = make_task(fam, rng, root, f"{fam.name}-test-0000", "test")
            tasks.append(t)
            # Verify decoupled keys (no bare 'id' column)
            for stem, fname in t["files"].items():
                if not fname.endswith(".tif"):
                    layers, _ = read_workspace(root / t["variants"]["low"]["dir"], {stem: fname})
                    assert "id" not in layers[stem].columns, f"Bare 'id' column found in {fam.name}/{stem}"
            # Verify low variant is strictly within +/-1.5 deg (or +/-2.5 deg for large)
            lat_bound = 2.51 if fam.large else 1.51
            assert abs(t["variants"]["low"]["lat"]) <= lat_bound

        (root / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
        table = audit_dataset(root)

        for fam in FAMILIES:
            row = table[fam.name]
            assert row["reference_pass"] == 1, f"Reference failed on {fam.name}"
            assert row["utm_pass"] == 0, f"Naive UTM unexpectedly passed {fam.name}"
            if fam.group == "composite_vector":
                assert row["wm_low_pass"] == 1, f"EPSG:3857 should pass low on {fam.name}"
                assert row["webmercator_pass"] == 0, f"EPSG:3857 should fail 4-loc oracle on {fam.name}"
            if fam.has_raster:
                assert row["wm_low_pass"] == 1, f"EPSG:3857 should pass low on raster {fam.name}"
                assert row["webmercator_pass"] == 0, f"EPSG:3857 should fail polar/high on raster {fam.name}"
                assert row["naive_nodata_caught"] == 1, f"Naive nodata reader should fail {fam.name}"


def test_metamorphic_perturbation_preserves_truth_on_raster_and_antimeridian():
    rng = np.random.default_rng(13)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for fam_name in ("buffer_exclusion_area", "raster_threshold_area", "raster_slope_exceed_area", "raster_zonal_mean"):
            fam = BY_NAME[fam_name]
            t = make_task(fam, rng, root, f"{fam_name}-test-0000", "test")
            low_dir = root / t["variants"]["low"]["dir"]
            low_truth = t["variants"]["low"]["truth"]

            for flip in (False, True):
                pdir = root / f"pert_{fam_name}_{int(flip)}"
                meta = perturb_workspace(low_dir, pdir, target_lat=58.3, target_lon=6.0, flip_gpkg_crs=flip, seed=42)
                assert meta["files"] == len(t["files"])
                layers, rasters = read_workspace(pdir, t["files"])
                crs_obj = reference_aeqd_crs(layers, rasters)
                val = fam.solve(layers, rasters, lambda g: g.to_crs(crs_obj), crs_obj, t["params"])
                # Allow 1.5% on double-warped discrete slope/threshold rasters
                rel = abs(val - low_truth) / max(abs(low_truth), 1e-12)
                assert rel <= 0.015, f"{fam_name} flip={flip}: orig={low_truth}, pert={val} (rel={rel:.4f})"


def test_multimodal_verifier_certification_accepts_raster_nodata_and_rejects_corrupted():
    rng = np.random.default_rng(21)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fam = BY_NAME["raster_zonal_mean"]
        t = make_task(fam, rng, root, "rzm-test-0000", "test")
        low_dir = root / t["variants"]["low"]["dir"]

        # A fragile script that hardcodes `arr != -9999` (passes low where nodata=-9999, fails when nodata flips!)
        fragile_script = (
            "import geopandas as gpd, numpy as np, rasterio, rasterio.features, shapely\n"
            "import glob\n"
            "zf = [f for f in glob.glob('zones.*') if not f.endswith('.tif')][0]\n"
            "z = gpd.read_file(zf)\n"
            f"sel = z[z['landuse'] == {t['params']['landuse']!r}]\n"
            "with rasterio.open('dem.tif') as src:\n"
            "    sel = sel.to_crs(src.crs)\n"
            "    arr = src.read(1).astype(float)\n"
            "    u = shapely.union_all(list(sel.geometry))\n"
            "    m = rasterio.features.geometry_mask([u], out_shape=arr.shape, transform=src.transform, invert=True)\n"
            "    vals = arr[m & (arr != -9999.0)]\n"
            "    print(float(np.mean(vals)))\n"
        )
        passing_cases = [(low_dir, fragile_script, t["variants"]["low"]["truth"], "real")]

        # 1. Valid nodata-sentinel recoding probe -> MUST certify!
        good_nodata_probe = '''
import shutil
from pathlib import Path
import numpy as np
import rasterio

def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() == ".tif":
            with rasterio.open(p) as src:
                arr = src.read(1)
                old_nd = float(src.nodata) if src.nodata is not None else -9999.0
                new_nd = -32768.0 if abs(old_nd + 9999.0) < 1e-3 else -9999.0
                out = np.where(np.isclose(arr, old_nd), np.float32(new_nd), arr)
                profile = src.profile.copy()
                profile.update(nodata=new_nd)
                with rasterio.open(dst_dir / p.name, "w", **profile) as dst:
                    dst.write(out, 1)
        else:
            shutil.copy2(p, dst_dir / p.name)
    return {"mode": "nodata_sentinel_recode"}
'''
        res_ok = certify_perturbation_operator(good_nodata_probe, [low_dir], passing_cases)
        assert res_ok["certified"], f"Expected valid nodata recode probe certified, got: {res_ok}"
        assert res_ok["metrics"]["caught"] == 1

        # 2. Corrupted raster probe (scales raster pixel values by 2x) -> MUST be rejected at conservation!
        bad_scale_probe = '''
import shutil
from pathlib import Path
import numpy as np
import rasterio

def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() == ".tif":
            with rasterio.open(p) as src:
                arr = src.read(1)
                nd = float(src.nodata) if src.nodata is not None else -9999.0
                out = np.where(np.isclose(arr, nd), arr, arr * np.float32(2.0))
                with rasterio.open(dst_dir / p.name, "w", **src.profile) as dst:
                    dst.write(out, 1)
        else:
            shutil.copy2(p, dst_dir / p.name)
    return {"mode": "bad_value_scale"}
'''
        res_bad = certify_perturbation_operator(bad_scale_probe, [low_dir], passing_cases)
        assert not res_bad["certified"] and res_bad["stage"] == "conservation"

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.gsb.audit import audit_task
from benchmarks.gsb.build import make_task
from benchmarks.gsb.families import FAMILIES
from benchmarks.gsb.harness import score

METRIC_REAL = {"buffer_union_area", "mean_nn_distance", "nearest_facility_max", "area_by_class",
               "perimeter_by_class", "overlay_area", "centroid_distance", "road_length_by_type",
               "density_in_polygon", "road_buffer_area", "max_pair_distance", "roads_in_district",
               "large_region_area", "large_route_length"}
LARGE = {"large_region_area", "large_route_length"}


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(1)
    ts = [make_task(f, rng, root, f"{f.name}-{i}", "test") for f in FAMILIES for i in range(3)]
    return root, ts


def test_prompt_hides_location(tasks):
    root, ts = tasks
    for t in ts:
        assert "EPSG" not in t["prompt"] and "latitude" not in t["prompt"]


def test_truth_consistent_across_variants(tasks):
    # same local geometry everywhere, so a correct answer barely moves between locations
    root, ts = tasks
    for t in ts:
        if t["family"] in LARGE:  # shifted in degrees, so the true answer changes with latitude
            continue
        vals = [v["truth"] for v in t["variants"].values()]
        assert max(vals) - min(vals) <= 0.003 * max(abs(max(vals)), 1e-9) + 1e-9, (t["task_id"], vals)


def test_utm_passes_and_webmercator_caught(tasks):
    root, ts = tasks
    for t in ts:
        r = audit_task(t, root)
        assert r["reference"] == "correct", (t["task_id"], r)
        if t["family"] not in LARGE:
            assert r["utm"] == "correct", (t["task_id"], r)
        if t["family"] in METRIC_REAL:
            assert r["webmercator"] != "correct", (t["task_id"], r)


CORRECT = '''
import geopandas as gpd
g = gpd.read_file("{f}")
g = g.to_crs(g.estimate_utm_crs())
print(g.buffer({r}).union_all().area * {k})
'''
WEBMERC = CORRECT.replace("g.to_crs(g.estimate_utm_crs())", 'g.to_crs("EPSG:3857")')


def test_harness_end_to_end(tasks):
    root, ts = tasks
    t = next(t for t in ts if t["family"] == "buffer_union_area" and not t["files"][t["params"]["stem"]].endswith(".csv"))
    k = {"m2": 1, "km2": 1e-6, "ha": 1e-4}[t["params"]["unit"]]
    args = dict(f=t["files"][t["params"]["stem"]], r=t["params"]["r"], k=k)
    assert score(t, CORRECT.format(**args), root)["category"] == "correct"
    assert score(t, WEBMERC.format(**args), root)["category"] in ("silent_wrong", "lucky")
    assert score(t, "raise SystemExit(1)", root)["category"] == "error"
    assert score(t, "print('area: 5')", root)["category"] == "error"


def test_equal_area_passes_large_area(tasks):
    from benchmarks.gsb.build import read_layers
    from benchmarks.gsb.families import BY_NAME
    from benchmarks.gsb.harness import matches
    root, ts = tasks
    fam = BY_NAME["large_region_area"]
    for t in (t for t in ts if t["family"] == "large_region_area"):
        for info in t["variants"].values():
            val = fam.solve(read_layers(root / info["dir"], t["files"]), lambda g: g.to_crs("EPSG:6933"), t["params"])
            assert matches(val, info["truth"], "real"), (t["task_id"], val, info["truth"])

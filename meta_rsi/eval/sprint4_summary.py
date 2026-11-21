"""Sprint 4/5 Multi-Seed Statistical Aggregator & Verifier Audit (`sprint4_summary.py`).

Aggregates 6-seed GSB v2 (`eval450`), 3-seed SciUnits (`eval280`), and real OSM (`n=24`)
evaluation JSONs, strictly verifying `usage.errors == 0` on every record, and computing
Mean ± SD, 95% bootstrap CIs, paired t-tests, and exact sign-flip permutation p-values.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from scipy import stats

GSB2_GROUPS: dict[str, set[str]] = {
    "CompV": {
        "area_weighted_pop_filter", "buffer_exclusion_area", "clipped_road_density",
        "covered_pop_dedup", "multi_ring_density", "nearest_service_gap",
        "dual_buffer_overlap_pop", "weighted_accessibility_ratio",
    },
    "Rast": {
        "raster_zonal_mean", "raster_threshold_area", "raster_slope_exceed_area",
        "raster_buffer_extract", "raster_pop_exposure",
        "raster_corridor_elevation", "raster_hotspot_in_buffer",
    },
    "Large": {"large_region_area_v2", "large_route_length_v2", "large_max_geodesic_span"},
    "Ctrl": {"district_mean_zero_trap_v2", "topological_containment_v2"},
}


def extract_metrics(path: Path) -> dict:
    d = json.loads(path.read_text())
    uerr = int((d.get("usage") or {}).get("errors", 0))
    a, s, h = d["ALL"], d["SEEN_18"], d["HELDOUT_3"]
    pf = d.get("per_family") or {}
    groups = {}
    for gname, fset in GSB2_GROUPS.items():
        tot_n = sum(pf[f]["n"] for f in fset if f in pf)
        tot_c = sum(pf[f]["n"] * pf[f]["correct"] for f in fset if f in pf)
        if tot_n > 0:
            groups[gname] = tot_c / tot_n
    return {
        "path": str(path),
        "n": a["n"],
        "all": float(a["correct"]),
        "seen": float(s["correct"]),
        "ho": float(h["correct"]),
        "lucky": float(a["lucky"]),
        "silent_wrong": float(a["silent_wrong"]),
        "error": float(a["error"]),
        "calls": float(d.get("avg_llm_calls", 0.0)),
        "uerr": uerr,
        "groups": groups,
    }


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def bootstrap_ci95(xs: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return xs[0], xs[0]
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def paired_tests(a: list[float], b: list[float]) -> dict:
    """Paired statistical tests for `a - b` (mean delta, 95% CI, paired t, exact sign-flip p)."""
    assert len(a) == len(b) and len(a) >= 2
    diffs = [x - y for x, y in zip(a, b)]
    m, s = mean_sd(diffs)
    lo, hi = bootstrap_ci95(diffs)
    if s < 1e-12:
        t_stat, t_p = (math.inf if m > 0 else (-math.inf if m < 0 else 0.0)), (0.0 if abs(m) > 0 else 1.0)
    else:
        t_res = stats.ttest_rel(a, b)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
    # Exact two-sided paired sign-flip permutation p-value (exact for n <= 12)
    n = len(diffs)
    obs = abs(sum(diffs))
    ge = 0
    total = 1 << n
    for mask in range(total):
        val = 0.0
        for i in range(n):
            val += diffs[i] if ((mask >> i) & 1) else -diffs[i]
        if abs(val) >= obs - 1e-12:
            ge += 1
    perm_p = ge / total
    try:
        w_res = stats.wilcoxon(a, b, zero_method="wilcox")
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = float("nan")
    return {
        "n": n,
        "mean_delta": m,
        "sd_delta": s,
        "ci95": [lo, hi],
        "t_stat": t_stat,
        "t_p": t_p,
        "wilcoxon_p": w_p,
        "exact_perm_p": perm_p,
    }


def _find_first(roots: list[Path], candidates: list[str]) -> Path | None:
    for cand in candidates:
        for r in roots:
            p = r / cand
            if p.exists():
                return p
    return None


def summarize_gsb2(runs_dir: Path) -> dict:
    roots = [
        runs_dir / "v7_s4_wave1c_g4b",
        runs_dir / "v7_s4_wave1c_g4a",
        runs_dir / "v7_s4_wave1b_g4b",
        runs_dir / "v7_s4_wave1b_g4a",
        runs_dir / "v7_s4_wave1a_g4b",
        runs_dir / "v7_s4_wave1a_g4a",
        runs_dir / "v6_gsb_v2_s3b_stage2",
        runs_dir / "v6_gsb_v2_s3b",
        runs_dir / "v6_gsb_v2_s3",
    ]
    methods: dict[str, dict[int, dict]] = {
        "darwinx_soft": {},
        "llm_judge": {},
        "meta_T0": {},
        "coev_tip_only": {},
        "arch_coev": {},
    }
    for s in range(6):
        p_dx = _find_first(
            roots,
            [f"eval_v2_s{s}_darwinx_soft.json"]
            + (["eval450_v2_sym26b_s0_darwinx_soft.json"] if s == 0 else []),
        )
        if p_dx:
            methods["darwinx_soft"][s] = extract_metrics(p_dx)
        p_lj = _find_first(roots, [f"eval_v2_s{s}_llm_judge.json"])
        if p_lj:
            methods["llm_judge"][s] = extract_metrics(p_lj)
        p_t0 = _find_first(roots, [f"eval_v2_s{s}_meta_T0.json"] + (["eval450_stage1_meta_T0.json"] if s == 0 else []))
        if p_t0:
            methods["meta_T0"][s] = extract_metrics(p_t0)
        cands_tip = (
            ["eval450_stage2_coev.json"]
            if s == 0
            else [f"eval_v2_s{s}_coev_v2.json", f"eval_v2_s{s}_coev.json"]
        )
        p_tip = _find_first(roots, cands_tip)
        if p_tip:
            methods["coev_tip_only"][s] = extract_metrics(p_tip)
        p_arch = _find_first(roots, [f"eval_v2_s{s}_arch_coev.json"])
        if p_arch:
            methods["arch_coev"][s] = extract_metrics(p_arch)

    singles = {}
    for label, cands in [
        ("seed_26b", ["seed_26b_nothink_v2.json"]),
        ("handcrafted_26b", ["handcrafted_26b_nothink_v2.json"]),
        ("runtime_meta_26b", ["eval_runtime_meta_26b_v2.json"]),
        ("s0_naive", ["eval_v2_s0_naive.json"]),
        ("s0_meta_gate_only", ["eval_v2_s0_meta_gate_only.json"]),
        ("s0_oracle_soft", ["eval_v2_s0_oracle_soft.json"]),
    ]:
        p = _find_first(roots, cands)
        if p:
            singles[label] = extract_metrics(p)

    # Check zero usage errors across all found files
    bad_uerr = []
    for mname, smap in methods.items():
        for s, rec in smap.items():
            if rec["uerr"] > 0:
                bad_uerr.append((f"{mname}:s{s}", rec["uerr"], rec["path"]))
    for sname, rec in singles.items():
        if rec["uerr"] > 0:
            bad_uerr.append((sname, rec["uerr"], rec["path"]))

    agg = {}
    for mname, smap in methods.items():
        seeds = sorted(smap)
        if not seeds:
            continue
        def _arr(key: str) -> list[float]:
            return [smap[s][key] for s in seeds]
        agg[mname] = {
            "seeds": seeds,
            "per_seed_all": {str(s): round(100 * smap[s]["all"], 2) for s in seeds},
            "all": mean_sd(_arr("all")),
            "all_ci95": bootstrap_ci95(_arr("all")),
            "seen": mean_sd(_arr("seen")),
            "ho": mean_sd(_arr("ho")),
            "lucky": mean_sd(_arr("lucky")),
            "error": mean_sd(_arr("error")),
            "calls": mean_sd(_arr("calls")),
            "groups": {
                g: mean_sd([smap[s]["groups"][g] for s in seeds if g in smap[s]["groups"]])
                for g in GSB2_GROUPS
            },
        }

    comparisons = {}
    primary = "arch_coev" if len(methods["arch_coev"]) >= 2 else "coev_tip_only"
    for base_m in ("darwinx_soft", "llm_judge", "meta_T0", "coev_tip_only"):
        if base_m == primary:
            continue
        shared = sorted(set(methods[primary]) & set(methods[base_m]))
        if len(shared) >= 2:
            comparisons[f"{primary}_vs_{base_m}"] = paired_tests(
                [methods[primary][s]["all"] for s in shared],
                [methods[base_m][s]["all"] for s in shared],
            )
            comparisons[f"{primary}_vs_{base_m}_ho"] = paired_tests(
                [methods[primary][s]["ho"] for s in shared],
                [methods[base_m][s]["ho"] for s in shared],
            )

    # SciUnits 3-seed paired matrix (prefer fixed v2 digest runs when present)
    sci_methods: dict[str, dict[int, dict]] = {"darwinx_soft": {}, "meta_soft": {}}
    for s in range(3):
        p_sd = _find_first(roots, [f"eval_sci_s{s}_darwinx_v2.json", f"eval_sci_s{s}_darwinx_soft.json"])
        if p_sd:
            sci_methods["darwinx_soft"][s] = extract_metrics(p_sd)
            if sci_methods["darwinx_soft"][s]["uerr"] > 0:
                bad_uerr.append((f"sci_darwinx_soft:s{s}", sci_methods["darwinx_soft"][s]["uerr"], str(p_sd)))
        p_sm = _find_first(roots, [f"eval_sci_s{s}_meta_v2.json", f"eval_sci_s{s}_meta_soft.json"])
        if p_sm:
            sci_methods["meta_soft"][s] = extract_metrics(p_sm)
            if sci_methods["meta_soft"][s]["uerr"] > 0:
                bad_uerr.append((f"sci_meta_soft:s{s}", sci_methods["meta_soft"][s]["uerr"], str(p_sm)))

    sci_agg = {}
    for gm, smap in sci_methods.items():
        seeds = sorted(smap)
        if not seeds:
            continue
        sci_agg[gm] = {
            "seeds": seeds,
            "per_seed_all": {str(s): round(100 * smap[s]["all"], 2) for s in seeds},
            "all": mean_sd([smap[s]["all"] for s in seeds]),
            "seen": mean_sd([smap[s]["seen"] for s in seeds]),
            "ho": mean_sd([smap[s]["ho"] for s in seeds]),
            "lucky": mean_sd([smap[s]["lucky"] for s in seeds]),
            "error": mean_sd([smap[s]["error"] for s in seeds]),
            "calls": mean_sd([smap[s]["calls"] for s in seeds]),
        }
    sci_shared = sorted(set(sci_methods["meta_soft"]) & set(sci_methods["darwinx_soft"]))
    if len(sci_shared) >= 2:
        comparisons["sci_meta_vs_darwinx"] = paired_tests(
            [sci_methods["meta_soft"][s]["all"] for s in sci_shared],
            [sci_methods["darwinx_soft"][s]["all"] for s in sci_shared],
        )

    return {
        "usage_error_violations": bad_uerr,
        "gsb2_multiseed": agg,
        "gsb2_singles": singles,
        "sci_multiseed": sci_agg,
        "paired_tests": comparisons,
    }


def _fmt(ms: tuple[float, float], pct: bool = True) -> str:
    m, s = ms
    return f"{100*m:5.1f} ± {100*s:4.1f}%" if pct else f"{m:4.2f} ± {s:4.2f}"


def print_report(rep: dict) -> None:
    if rep["usage_error_violations"]:
        print("WARNING: Non-zero usage errors detected:", rep["usage_error_violations"])
    else:
        print("AUDIT OK: usage.errors == 0 across ALL loaded evaluation JSONs.")

    print("\n=== GSB v2 Multi-Seed Matrix (eval450) ===")
    print(f"{'Method':<16} {'N':>2} | {'ALL':>13} {'Seen':>13} {'HO':>13} {'Lucky':>13} {'Err':>13} | {'CompV':>13} {'Rast':>13} {'Calls':>11}")
    print("-" * 136)
    for m, d in rep["gsb2_multiseed"].items():
        g = d["groups"]
        print(
            f"{m:<16} {len(d['seeds']):>2} | {_fmt(d['all']):>13} {_fmt(d['seen']):>13} {_fmt(d['ho']):>13} "
            f"{_fmt(d['lucky']):>13} {_fmt(d['error']):>13} | {_fmt(g['CompV']):>13} {_fmt(g['Rast']):>13} {_fmt(d['calls'], False):>11}"
        )
        print(f"   per-seed ALL: {d['per_seed_all']}")

    if rep["gsb2_singles"]:
        print("\n=== GSB v2 Reference / Ablation Runs (eval450) ===")
        for label, r in rep["gsb2_singles"].items():
            g = r["groups"]
            print(
                f"{label:<20} | ALL={100*r['all']:5.1f}% Seen={100*r['seen']:5.1f}% HO={100*r['ho']:5.1f}% "
                f"Lucky={100*r['lucky']:5.1f}% Err={100*r['error']:5.1f}% Calls={r['calls']:.2f} | "
                f"CompV={100*g.get('CompV',0):5.1f}% Rast={100*g.get('Rast',0):5.1f}% Large={100*g.get('Large',0):5.1f}% Ctrl={100*g.get('Ctrl',0):5.1f}%"
            )

    if rep["sci_multiseed"]:
        print("\n=== SciUnits Multi-Seed Matrix (eval280) ===")
        for m, d in rep["sci_multiseed"].items():
            print(
                f"sci_{m:<12} {len(d['seeds']):>2} | ALL={_fmt(d['all'])} Seen={_fmt(d['seen'])} HO={_fmt(d['ho'])} "
                f"Lucky={_fmt(d['lucky'])} Err={_fmt(d['error'])} Calls={_fmt(d['calls'], False)} | {d['per_seed_all']}"
            )

    if rep["paired_tests"]:
        print("\n=== Paired Statistical Tests ===")
        for cname, t in rep["paired_tests"].items():
            print(
                f"{cname:<28} (n={t['n']}) | delta={100*t['mean_delta']:+5.1f} ± {100*t['sd_delta']:4.1f} pp "
                f"(95% CI [{100*t['ci95'][0]:+5.1f}, {100*t['ci95'][1]:+5.1f}]) | "
                f"t={t['t_stat']:.2f} (p={t['t_p']:.4g}) | perm_p={t['exact_perm_p']:.4g}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Sprint 4/5 multi-seed evaluation JSONs")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    rep = summarize_gsb2(Path(a.runs_dir))
    print_report(rep)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

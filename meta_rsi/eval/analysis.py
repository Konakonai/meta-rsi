"""Publication-grade statistical analysis of multi-seed RSI evolution + unseen test eval."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy import stats


def _ms(vals: list[float]) -> tuple[float, float]:
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    return m, s


def print_ood_report(ood_dir: Path, seeds: list[int], gates: list[str], gate_labels: dict[str, str]) -> None:
    seed_p = ood_dir / "seed_e4b_nothink.json"
    if not seed_p.exists():
        return
    seed_ev = json.loads(seed_p.read_text())
    n_loc = seed_ev["ALL_LOCAL"]["n"]
    n_sn = seed_ev["SEEN_LOCAL_16"]["n"]
    n_ho = seed_ev["HELDOUT_3"]["n"]
    print(f"\n=== POLAR OOD STRESS TEST (>84N / <80S, n={n_loc} local tasks: {n_sn} seen + {n_ho} held-out) ===")
    print(
        f"{'Exp':<20} | {'ALL_Corr':>8} {'Sn_Corr':>7} {'HO_Corr':>7} | "
        f"{'Lucky':>7} {'SilW':>7} {'Error':>7} | {'Swallow':>8} {'Calls':>5}"
    )
    print("-" * 92)

    def _fmt_sw(ev: dict) -> str:
        sw = ev.get("swallowed_count")
        n = max(ev["ALL_LOCAL"]["n"], 1)
        return "    -   " if sw is None else f"{sw:3d}({sw/n*100:3.0f}%)"

    s_all = seed_ev["ALL_LOCAL"]
    print(
        f"{'Gen-0 Seed':<20} | "
        f"{s_all['correct']*100:7.1f}% {seed_ev['SEEN_LOCAL_16']['correct']*100:6.1f}% {seed_ev['HELDOUT_3']['correct']*100:6.1f}% | "
        f"{s_all['lucky']*100:6.1f}% {s_all['silent_wrong']*100:6.1f}% {s_all['error']*100:6.1f}% | "
        f"{_fmt_sw(seed_ev):>8} {seed_ev['avg_llm_calls']:5.2f}"
    )

    by_gate: dict[str, list[dict]] = defaultdict(list)
    for g in gates:
        for s in seeds:
            exp = f"v3_gis_s{s}_{g}"
            ev_p = ood_dir / f"{exp}.json"
            if not ev_p.exists():
                continue
            ev = json.loads(ev_p.read_text())
            a = ev["ALL_LOCAL"]
            sw = ev.get("swallowed_count")
            row = {
                "seed": s,
                "exp": exp,
                "all_acc": a["correct"],
                "seen_acc": ev["SEEN_LOCAL_16"]["correct"],
                "ho_acc": ev["HELDOUT_3"]["correct"],
                "lucky": a["lucky"],
                "sw": a["silent_wrong"],
                "err": a["error"],
                "swallowed_rate": (sw / max(a["n"], 1)) if sw is not None else None,
                "calls": ev["avg_llm_calls"],
            }
            by_gate[g].append(row)
            print(
                f"{exp:<20} | "
                f"{row['all_acc']*100:7.1f}% {row['seen_acc']*100:6.1f}% {row['ho_acc']*100:6.1f}% | "
                f"{row['lucky']*100:6.1f}% {row['sw']*100:6.1f}% {row['err']*100:6.1f}% | "
                f"{_fmt_sw(ev):>8} {row['calls']:5.2f}"
            )

    print(f"\n=== POLAR OOD 4-SEED AGGREGATE (n={n_loc}, mean ± std) ===")
    print(
        f"{'Method':<22} {'N':>2} | {'ALL_Corr':>11} {'HO_Corr':>11} | "
        f"{'Lucky':>11} {'SilW':>11} {'Error':>11} | {'Swallow%':>11}"
    )
    print("-" * 105)
    for g in gates:
        rows = by_gate[g]
        if not rows:
            continue
        def fm(k):
            m, s = _ms([r[k] for r in rows])
            return f"{m*100:4.1f}±{s*100:4.1f}%"
        sw_vals = [r["swallowed_rate"] for r in rows if r["swallowed_rate"] is not None]
        if sw_vals:
            sm, ss = _ms(sw_vals)
            sw_s = f"{sm*100:4.1f}±{ss*100:4.1f}%"
        else:
            sw_s = "     -     "
        print(
            f"{gate_labels[g]:<22} {len(rows):>2} | "
            f"{fm('all_acc'):>11} {fm('ho_acc'):>11} | "
            f"{fm('lucky'):>11} {fm('sw'):>11} {fm('err'):>11} | {sw_s:>11}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistical analysis of multi-seed RSI evolution + unseen test / OOD eval")
    ap.add_argument("eval_dir", nargs="?", default="runs/eval150", help="Directory with unseen test JSONs (default: runs/eval150)")
    ap.add_argument("--ood-dir", default="", help="Optional directory with Polar OOD eval JSONs (e.g. runs/eval_ood)")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    root = Path("runs")
    seeds = [0, 1, 2, 3]
    gates = ["naive", "preserve", "meta"]
    gate_labels = {"naive": "naive", "preserve": "preserve (DarwinX)", "meta": "metamorphic (Ours)"}

    seed_base = json.loads((eval_dir / "seed_e4b_nothink.json").read_text())
    n_all = seed_base["ALL"]["n"]
    n_seen = seed_base["SEEN_18"]["n"]
    n_ho = seed_base["HELDOUT_3"]["n"]
    print(f"=== GEN-0 SEED BASELINE ON {n_all}-TASK UNSEEN TEST SET (E4B no-think, temp=0) ===")
    for k in ("ALL", "SEEN_18", "HELDOUT_3"):
        r = seed_base[k]
        print(f"  {k:<10}: correct={r['correct']*100:5.1f}%  lucky={r['lucky']*100:5.1f}%  silent_wrong={r['silent_wrong']*100:5.1f}%  error={r['error']*100:5.1f}%")

    print(f"\n=== PER-SEED RESULTS (DEV 36 vs {n_all}-TASK UNSEEN TEST) ===")
    print(
        f"{'Exp':<20} {'Best':>4} | {'DevLow':>6} {'OraDev':>6} {'DevGap':>7} | "
        f"{f'T{n_all}_ALL':>8} {f'T{n_seen}_Sn':>7} {f'T{n_ho}_HO':>7} | {'T_Lucky':>7} {'T_SilW':>7} {'T_Err':>6} {'Calls':>5}"
    )
    print("-" * 116)

    by_gate: dict[str, list[dict]] = defaultdict(list)
    for g in gates:
        for s in seeds:
            exp = f"v3_gis_s{s}_{g}"
            arch_p = root / exp / "archive.json"
            ev_p = eval_dir / f"{exp}.json"
            if not (arch_p.exists() and ev_p.exists()):
                continue
            arch = json.loads(arch_p.read_text())
            best = max(arch["entries"], key=lambda e: (e["in_loop_rate"], -e["dev_error_rate"], e["gen"]))
            ev = json.loads(ev_p.read_text())
            row = {
                "seed": s,
                "exp": exp,
                "best_gen": best["gen"],
                "dev_low": best["dev_low_rate"],
                "ora_dev": best["oracle_dev_rate"],
                "dev_gap": best["dev_low_rate"] - best["oracle_dev_rate"],
                "all_acc": ev["ALL"]["correct"],
                "seen_acc": ev["SEEN_18"]["correct"],
                "ho_acc": ev["HELDOUT_3"]["correct"],
                "all_lucky": ev["ALL"]["lucky"],
                "all_sw": ev["ALL"]["silent_wrong"],
                "all_err": ev["ALL"]["error"],
                "calls": ev["avg_llm_calls"],
                "per_family": ev["per_family"],
            }
            by_gate[g].append(row)
            print(
                f"{exp:<20} g{row['best_gen']:02d}  | "
                f"{row['dev_low']*100:5.1f}% {row['ora_dev']*100:5.1f}% {row['dev_gap']*100:+6.1f}% | "
                f"{row['all_acc']*100:7.1f}% {row['seen_acc']*100:6.1f}% {row['ho_acc']*100:6.1f}% | "
                f"{row['all_lucky']*100:6.1f}% {row['all_sw']*100:6.1f}% {row['all_err']*100:5.1f}% {row['calls']:5.2f}"
            )

    print(f"\n=== 4-SEED AGGREGATE ON {n_all}-TASK UNSEEN TEST SET (mean ± std) ===")
    print(
        f"{'Method':<22} {'N':>2} | {'DevLow':>11} {'OraDev':>11} {'DevGap':>11} | "
        f"{f'T{n_all}_ALL':>11} {f'T{n_seen}_Seen':>11} {f'T{n_ho}_HO':>11} | {'T_Lucky':>11} {'T_Err':>11} {'Calls':>9}"
    )
    print("-" * 145)
    for g in gates:
        rows = by_gate[g]
        if not rows:
            continue
        def fm(k):
            m, s = _ms([r[k] for r in rows])
            return f"{m*100:4.1f}±{s*100:4.1f}%"
        cm, cs = _ms([r["calls"] for r in rows])
        print(
            f"{gate_labels[g]:<22} {len(rows):>2} | "
            f"{fm('dev_low'):>11} {fm('ora_dev'):>11} {fm('dev_gap'):>11} | "
            f"{fm('all_acc'):>11} {fm('seen_acc'):>11} {fm('ho_acc'):>11} | "
            f"{fm('all_lucky'):>11} {fm('all_err'):>11} {cm:4.2f}±{cs:4.2f}"
        )

    # Paired comparisons across matched seeds
    if len(by_gate["meta"]) >= 3 and len(by_gate["preserve"]) >= 3:
        print(f"\n=== PAIRED SEED-BY-SEED COMPARISONS ({n_all}-TASK TEST SET) ===")
        for other in ("preserve", "naive"):
            m_map = {r["seed"]: r for r in by_gate["meta"]}
            o_map = {r["seed"]: r for r in by_gate[other]}
            common = sorted(set(m_map) & set(o_map))
            if len(common) < 2:
                continue
            for metric, label in [("all_acc", f"T{n_all} ALL"), ("ho_acc", f"T{n_ho} Held-out"), ("all_lucky", f"T{n_all} Lucky")]:
                a = [m_map[s][metric] for s in common]
                b = [o_map[s][metric] for s in common]
                diffs = [x - y for x, y in zip(a, b)]
                dm, ds = _ms(diffs)
                t_stat, p_val = stats.ttest_rel(a, b)
                wins = sum(1 for d in diffs if (d < 0 if "lucky" in metric else d > 0))
                print(
                    f"  meta vs {other:<8} | {label:<14}: delta={dm*100:+5.1f}±{ds*100:4.1f}%  "
                    f"wins={wins}/{len(common)}  paired_t={t_stat:+.2f}, p={p_val:.4f}"
                )

    # Category breakdown: Local projection-dependent vs Non-projection traps vs Large-extent geodesic
    non_proj_fams = {"points_in_polygon", "area_weighted_population", "covered_population", "mean_per_district"}
    large_fams = {"large_region_area", "large_route_length"}

    def _subset_acc(per_fam: dict[str, dict], pred) -> tuple[float, float, int]:
        tot_n = 0
        tot_c = 0.0
        tot_l = 0.0
        for fam, st in per_fam.items():
            if pred(fam):
                tot_n += st["n"]
                tot_c += st["correct"] * st["n"]
                tot_l += st["lucky"] * st["n"]
        return (tot_c / tot_n if tot_n else 0.0), (tot_l / tot_n if tot_n else 0.0), tot_n

    groups = [
        ("LOCAL_PROJ (15)", lambda f: f not in non_proj_fams and f not in large_fams),
        ("NON_PROJ_TRAP (4)", lambda f: f in non_proj_fams),
        ("LARGE_EXTENT (2)", lambda f: f in large_fams),
    ]
    print(f"\n=== BREAKDOWN BY TASK CATEGORY ON {n_all}-TASK TEST SET (correct% / lucky%) ===")
    hdr = f"{'Method':<22} | " + " | ".join(f"{name:>19}" for name, _ in groups)
    print(hdr)
    print("-" * len(hdr))
    seed_cells = []
    for _, pred in groups:
        c, l, _ = _subset_acc(seed_base["per_family"], pred)
        seed_cells.append(f"{c*100:5.1f}% / {l*100:4.1f}%     ")
    print(f"{'Gen-0 Seed':<22} | " + " | ".join(f"{s:>19}" for s in seed_cells))
    for g in gates:
        rows = by_gate[g]
        if not rows:
            continue
        cells = []
        for _, pred in groups:
            cs, ls = [], []
            for r in rows:
                c, l, _ = _subset_acc(r["per_family"], pred)
                cs.append(c)
                ls.append(l)
            cm, csd = _ms(cs)
            lm, _ = _ms(ls)
            cells.append(f"{cm*100:4.1f}±{csd*100:4.1f}% / {lm*100:4.1f}%")
        print(f"{gate_labels[g]:<22} | " + " | ".join(f"{s:>19}" for s in cells))

    # Zero-shot v1 reference comparison
    zdir = Path("results/zero_shot_v1")
    if zdir.exists():
        print("\n=== REFERENCE: ZERO-SHOT V1 BASELINES ON SAME 480 TEST TASKS ===")
        for name in ("e4b_think", "e4b_think_hint", "e4b_think_rules", "q27_low", "g26b_think"):
            p = zdir / f"{name}.summary.json"
            if p.exists():
                s = json.loads(p.read_text())["summary"]["ALL"]
                print(
                    f"  {name:<18}: pass@1={s['correct']*100:5.1f}%  lucky={s['lucky']*100:5.1f}%  "
                    f"silent_wrong={s['silent_wrong']*100:5.1f}%  error={s['error']*100:5.1f}%"
                )

    if args.ood_dir:
        ood_path = Path(args.ood_dir)
        if ood_path.exists():
            print_ood_report(ood_path, seeds, gates, gate_labels)


if __name__ == "__main__":
    main()

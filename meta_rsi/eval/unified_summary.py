"""Seed- and task-paired summary of a unified 6-seed re-evaluation (scripts/rebuttal/unified_6seed_eval.py).

Per method and protocol: mean +- sd over seeds of the share of tasks correct on every variant (Protocol A
averaged over its repeats), the lucky / silent-wrong / error shares, and the shift from the original eval.
Protocol A vs B: what live deployment (B) does to the tasks Protocol A scored correct, lucky or silent-wrong.
Each method against a reference (default arch_coev): the seed-paired difference, an exact sign-flip test over
seeds (Holm-adjusted across methods), and a 95% interval from a bootstrap that resamples seeds (paired) and
task families. Decision rule (docs/BLIND_PROTOCOL.md, Addendum 2): the only confirmatory comparison is
arch_coev - naive under Protocol A, established iff the exact sign-flip p < 0.05 (all 6 seeds agree); every other
comparison is descriptive, with t5 intervals on the seed-level differences (95% and Bonferroni over BONF_K) and the
seeds x families bootstrap as a descriptive interval.

    python -m meta_rsi.eval.unified_summary runs/v10_unified_6seed [--ref arch_coev]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as _st

CATS = ("correct", "lucky", "silent_wrong", "error")
CONFIRMATORY = ("arch_coev", "naive")  # (method, reference): the one pre-registered confirmatory comparison
BONF_K = 14  # primary-outcome comparisons: 7 methods vs naive + 7 vs arch_coev
SINGLE_CALL = {"naive", "meta_gate_only", "oracle_soft", "darwinx_soft", "meta_T0"}  # backend-check comparison group
_EVAL_RE = re.compile(r"^(?P<tag>.+)_(?P<proto>[AB])_rep(?P<rep>\d+)\.json$")


def _cat(c: str | None) -> str:
    return c if c in CATS else "error"  # no_code / missing count as error, as in eval_test's rates()


def load_runs(out: Path, with_reps: bool = False):
    """manifest, {(tag, protocol): [ {task_id: record} per clean repeat, in rep order ]} (and the rep numbers)."""
    manifest = json.loads((out / "manifest.json").read_text())
    tags = {e["tag"] for e in manifest["entries"]}
    found: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for f in sorted((out / "evals").glob("*.json")):
        m = _EVAL_RE.match(f.name)
        if not m or m["tag"] not in tags:
            continue
        d = json.loads(f.read_text())
        recs = d.get("records") or []
        if int((d.get("usage") or {}).get("errors", 1)) != 0 or not recs or len(recs) != d["ALL"]["n"]:
            continue  # unfinished or backend errors: `run` redoes these
        found[(m["tag"], m["proto"])].append((int(m["rep"]), {r["task_id"]: r for r in recs}))
    runs = {k: [recs for _, recs in sorted(v, key=lambda x: x[0])] for k, v in found.items()}
    if with_reps:
        return manifest, runs, {k: sorted(r for r, _ in v) for k, v in found.items()}
    return manifest, runs


def t_interval(diffs: list[float], level: float = 0.95) -> tuple[float, float]:
    """Two-sided t interval for the mean of seed-level paired differences (df = n - 1)."""
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2:
        return float("nan"), float("nan")
    half = _st.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean() - half), float(d.mean() + half)


def _share_correct(recs: dict[str, dict], tids) -> float:
    return float(np.mean([_cat(recs[t]["category"]) == "correct" for t in tids]))


def backend_check(entries: list[dict], runs: dict, rep_ids: dict) -> dict:
    """Pre-registered drop rule for reused Protocol A repeat 1 (Addendum 2, item 4)."""
    groups: dict[str, list[float]] = {"reused": [], "fresh_single_call": []}
    for e in entries:
        k = (e["tag"], "A")
        if rep_ids.get(k, [])[:2] != [1, 2]:
            continue
        r1, r2 = runs[k][0], runs[k][1]
        common = sorted(set(r1) & set(r2))
        g = _share_correct(r1, common) - _share_correct(r2, common)
        if e.get("reused_A_rep1"):
            groups["reused"].append(g)
        elif e["method"] in SINGLE_CALL:
            groups["fresh_single_call"].append(g)
    out = {k: {"n": len(v), "signed_mean": float(np.mean(v)) if v else None,
               "abs_mean": float(np.mean(np.abs(v))) if v else None} for k, v in groups.items()}
    ru, fr = out["reused"], out["fresh_single_call"]
    drop = bool(ru["n"] and fr["n"] and ((ru["abs_mean"] - fr["abs_mean"]) > 0.02 or abs(ru["signed_mean"]) > 0.02))
    out["drop_reused_rep1"] = drop
    return out


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation p-value for the mean of paired differences."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return float("nan")
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def hier_bootstrap_ci(x1: np.ndarray, x2: np.ndarray, fam: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(x1 - x2); x* are (n_seeds, n_tasks) paired by row (seed) and column (task).

    Resamples seeds with replacement (keeping the pairing), then task families with replacement (all tasks of a
    drawn family enter with it), so both evolution randomness and test-set composition widen the interval.
    """
    rng = np.random.default_rng(seed)
    diff = x1 - x2
    fams = np.unique(fam)
    onehot = (fam[None, :] == fams[:, None]).astype(float)  # (n_fams, n_tasks)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        rows = diff[rng.integers(0, diff.shape[0], diff.shape[0])]
        w = np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) @ onehot  # task weights
        stats[b] = (rows @ w).sum() / (rows.shape[0] * w.sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(out: Path, ref: str = "arch_coev", n_boot: int = 4000, drop_reused_rep1: bool | None = None) -> dict:
    manifest, runs, rep_ids = load_runs(out, with_reps=True)
    entries = manifest["entries"]
    bcheck = backend_check(entries, runs, rep_ids)
    drop = bcheck["drop_reused_rep1"] if drop_reused_rep1 is None else drop_reused_rep1
    if drop:  # every later analysis (A, A->B, repeats) then sees only repeats >= 2 of the reused harnesses
        for e in entries:
            k = (e["tag"], "A")
            if e.get("reused_A_rep1") and rep_ids.get(k, [None])[0] == 1:
                runs[k], rep_ids[k] = runs[k][1:], rep_ids[k][1:]
    methods = list(dict.fromkeys(e["method"] for e in entries))
    task_ids = sorted({tid for reps in runs.values() for recs in reps for tid in recs})
    fam_of = {tid: recs[tid]["family"] for reps in runs.values() for recs in reps for tid in recs}
    tidx = {t: i for i, t in enumerate(task_ids)}
    fam = np.array([fam_of[t] for t in task_ids])

    def matrix(method: str, proto: str, cat: str = "correct") -> dict[int, np.ndarray]:
        """{seed: per-task share of repeats in `cat`} for one method; seeds without a clean run are absent."""
        outm = {}
        for e in entries:
            reps = runs.get((e["tag"], proto)) if e["method"] == method else None
            if not reps:
                continue
            v = np.full(len(task_ids), np.nan)
            for tid in task_ids:
                vals = [_cat(r[tid]["category"]) == cat for r in reps if tid in r]
                if vals:
                    v[tidx[tid]] = np.mean(vals)
            outm[e["seed"]] = v
        return outm

    report: dict = {"n_tasks": len(task_ids), "n_families": int(len(np.unique(fam))), "methods": {}, "vs_ref": {},
                    "backend_check": bcheck | {"applied": drop}}
    b_tasks = sorted({t for (tag, proto), reps in runs.items() if proto == "B" for r in reps for t in r})
    b_mask = np.array([t in set(b_tasks) for t in task_ids])
    for proto in ("A", "B"):
        for m in methods:
            per_cat = {c: matrix(m, proto, c) for c in CATS}
            seeds = sorted(per_cat["correct"])
            if not seeds:
                continue
            row = {"seeds": seeds, "n_reps": {e["seed"]: len(runs.get((e["tag"], proto), [])) for e in entries if e["method"] == m}}
            for c in CATS:
                vals = [float(np.nanmean(per_cat[c][s])) for s in seeds]
                row[c] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "per_seed": vals}
            sil = [row["lucky"]["per_seed"][i] + row["silent_wrong"]["per_seed"][i] for i in range(len(seeds))]
            row["silent_failure"] = {"mean": float(np.mean(sil)), "sd": float(np.std(sil, ddof=1)) if len(sil) > 1 else 0.0}
            if proto == "A" and b_mask.any():  # is the 100-task Protocol B subset representative under Protocol A?
                row["A_on_B_subset"] = float(np.mean([np.nanmean(per_cat["correct"][s][b_mask]) for s in seeds]))
            if proto == "A":
                orig = {e["seed"]: e.get("orig_correct") for e in entries if e["method"] == m}
                shift = [row["correct"]["per_seed"][i] - orig[s] for i, s in enumerate(seeds) if orig.get(s) is not None]
                row["shift_vs_orig_eval"] = float(np.mean(shift)) if shift else None
                row["shift_per_seed"] = {s: row["correct"]["per_seed"][i] - orig[s]
                                         for i, s in enumerate(seeds) if orig.get(s) is not None}
                # Pre-registered seed-0 check (docs/BLIND_PROTOCOL.md): seed 0 minus the mean of the other seeds
                rest = [row["correct"]["per_seed"][i] for i, s in enumerate(seeds) if s != 0]
                rest_o = [orig[s] for s in seeds if s != 0 and orig.get(s) is not None]
                if 0 in seeds and rest:
                    row["seed0_gap"] = row["correct"]["per_seed"][seeds.index(0)] - float(np.mean(rest))
                    row["seed0_gap_orig"] = (orig[0] - float(np.mean(rest_o))) if orig.get(0) is not None and rest_o else None
                row["orig_models"] = sorted({str(e.get("orig_model")) for e in entries if e["method"] == m})
            report["methods"].setdefault(m, {})[proto] = row

        if ref in methods:
            ref_m = matrix(ref, proto)
            rows, ps = [], []
            for m in methods:
                if m == ref:
                    continue
                mm = matrix(m, proto)
                common = sorted(set(mm) & set(ref_m))
                if not common:
                    continue
                keep = ~np.isnan(np.stack([mm[s] for s in common] + [ref_m[s] for s in common])).any(axis=0)
                x1 = np.stack([mm[s][keep] for s in common])
                x2 = np.stack([ref_m[s][keep] for s in common])
                d = (x1.mean(axis=1) - x2.mean(axis=1)).tolist()
                lo, hi = hier_bootstrap_ci(x1, x2, fam[keep], n_boot=n_boot)
                conf = proto == "A" and (m, ref) in (CONFIRMATORY, CONFIRMATORY[::-1])
                rows.append({"method": m, "n_seeds": len(common), "diff": float(np.mean(d)), "per_seed_diff": d,
                             "sign_flip_p": sign_flip_p(d), "boot_ci95": [lo, hi],
                             "t_ci95": list(t_interval(d)), "t_ci_bonf": list(t_interval(d, 1 - 0.05 / BONF_K)),
                             "confirmatory": conf, "established": bool(conf and sign_flip_p(d) < 0.05)})
                ps.append(rows[-1]["sign_flip_p"])
            for r, p_adj in zip(rows, holm(ps)):
                r["sign_flip_p_holm"] = p_adj
            report["vs_ref"][proto] = {"ref": ref, "rows": rows}

    # Protocol A -> B: the live fate of each A category (first repeat of each, per seed and task).
    trans: dict[str, Counter] = {}
    for m in methods:
        cnt: Counter = Counter()
        for e in entries:
            if e["method"] != m or not runs.get((e["tag"], "A")) or not runs.get((e["tag"], "B")):
                continue
            ra, rb = runs[(e["tag"], "A")][0], runs[(e["tag"], "B")][0]
            for tid in set(ra) & set(rb):
                cnt[(_cat(ra[tid]["category"]), _cat(rb[tid]["category"]))] += 1
        if cnt:
            trans[m] = cnt
    report["a_to_b"] = {
        m: {a_c: {"n": sum(cnt[(a_c, b)] for b in CATS),
                  "b_correct": cnt[(a_c, "correct")] / max(sum(cnt[(a_c, b)] for b in CATS), 1)} for a_c in CATS}
        for m, cnt in trans.items()
    }

    # Repeat variability of Protocol A on the same harness and server.
    deltas, flips = [], []
    for e in entries:
        reps = runs.get((e["tag"], "A"), [])
        for r1, r2 in itertools.combinations(reps, 2):
            common = set(r1) & set(r2)
            c1 = [_cat(r1[t]["category"]) == "correct" for t in common]
            c2 = [_cat(r2[t]["category"]) == "correct" for t in common]
            deltas.append(abs(np.mean(c1) - np.mean(c2)))
            flips.append(np.mean([a != b for a, b in zip(c1, c2)]))
    report["repeat_variability_A"] = (
        {"pairs": len(deltas), "mean_abs_delta": float(np.mean(deltas)), "max_abs_delta": float(np.max(deltas)),
         "mean_task_flip_rate": float(np.mean(flips))} if deltas else None
    )

    # Pre-registered backend check: |rep1 - rep2| for harnesses whose rep 1 was reused from an earlier server,
    # against harnesses whose repeats both ran on the unified server.
    gaps: dict[str, list[float]] = {"reused": [], "fresh": []}
    for e in entries:
        reps = runs.get((e["tag"], "A"), [])
        if len(reps) >= 2:
            common = set(reps[0]) & set(reps[1])
            c = [np.mean([_cat(r[t]["category"]) == "correct" for t in common]) for r in reps[:2]]
            gaps["reused" if e.get("reused_A_rep1") else "fresh"].append(abs(c[0] - c[1]))
    report["rep_gap_A"] = {k: {"n": len(v), "mean_abs": float(np.mean(v)) if v else None} for k, v in gaps.items()}
    g3 = [report["methods"].get(m, {}).get("A", {}).get("seed0_gap") for m in ("naive", "meta_gate_only", "oracle_soft")]
    report["seed0_gap_mean_3"] = float(np.mean(g3)) if all(g is not None for g in g3) else None
    report["missing_harnesses"] = manifest.get("missing", [])
    return report


def render(rep: dict) -> str:
    pct = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    lines = [f"Unified re-evaluation: {rep['n_tasks']} tasks, {rep['n_families']} families", ""]
    for proto in ("A", "B"):
        lines += [f"## Protocol {proto}", "",
                  "| method | seeds | correct | lucky | silent_wrong | error |" + (" vs orig eval |" if proto == "A" else ""),
                  "|---|---|---|---|---|---|" + ("---|" if proto == "A" else "")]
        for m, by_p in rep["methods"].items():
            r = by_p.get(proto)
            if not r:
                continue
            cells = [f"{pct(r[c]['mean'])} ± {pct(r[c]['sd']).strip()}" for c in CATS]
            extra = ""
            if proto == "A":
                s = r.get("shift_vs_orig_eval")
                extra = f" {'n/a' if s is None else f'{100 * s:+.1f}'} |"
            lines.append(f"| {m} | {len(r['seeds'])} | " + " | ".join(cells) + " |" + extra)
        vs = rep["vs_ref"].get(proto)
        if vs:
            lines += ["", f"Difference in correct vs `{vs['ref']}` (seed-paired; descriptive unless marked confirmatory):", "",
                      f"| method | seeds | diff (pp) | sign-flip p | Holm p | t5 95% (pp) | t5 Bonferroni/{BONF_K} (pp) | bootstrap seeds×families (pp) |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in vs["rows"]:
                lo, hi = r["boot_ci95"]
                (tl, th), (bl, bh) = r["t_ci95"], r["t_ci_bonf"]
                tag = " **(confirmatory: " + ("established" if r["established"] else "not established") + ")**" if r["confirmatory"] else ""
                lines.append(f"| {r['method']}{tag} | {r['n_seeds']} | {100 * r['diff']:+.1f} | {r['sign_flip_p']:.3f} | "
                             f"{r['sign_flip_p_holm']:.3f} | [{100 * tl:+.1f}, {100 * th:+.1f}] | [{100 * bl:+.1f}, {100 * bh:+.1f}] | "
                             f"[{100 * lo:+.1f}, {100 * hi:+.1f}] |")
        lines.append("")
    if rep["a_to_b"]:
        lines += ["## Protocol A category -> share correct under live deployment (B)", "",
                  "| method | A correct | A lucky | A silent_wrong | A error |", "|---|---|---|---|---|"]
        for m, t in rep["a_to_b"].items():
            lines.append(f"| {m} | " + " | ".join(f"{pct(t[c]['b_correct'])} (n={t[c]['n']})" for c in CATS) + " |")
        lines.append("")
    gap_rows = [(m, by_p["A"]) for m, by_p in rep["methods"].items() if "seed0_gap" in by_p.get("A", {})]
    if gap_rows:
        lines += ["## Seed 0 minus the mean of seeds 1-5 (Protocol A correct, pp)", "",
                  "| method | unified | original eval | per-seed shift vs original |", "|---|---|---|---|"]
        for m, r in gap_rows:
            o = r.get("seed0_gap_orig")
            sh = " / ".join(f"s{k}:{100 * v:+.1f}" for k, v in r["shift_per_seed"].items())
            lines.append(f"| {m} | {100 * r['seed0_gap']:+.1f} | {'n/a' if o is None else f'{100 * o:+.1f}'} | {sh} |")
        lines.append("")
    if rep.get("seed0_gap_mean_3") is not None:
        lines.append(f"Seed-0 gap, mean over naive / meta_gate_only / oracle_soft: {100 * rep['seed0_gap_mean_3']:+.1f} pp")
    bc = rep.get("backend_check") or {}
    if bc:
        lines.append("Backend check (rep 1 - rep 2): " + ", ".join(
            f"{k} n={v['n']} signed {100 * v['signed_mean']:+.1f} abs {100 * v['abs_mean']:.1f} pp"
            for k, v in bc.items() if isinstance(v, dict) and v.get("n")) + f"; drop reused rep 1: {bc.get('applied')}")
    sub = [(m, by_p["A"]) for m, by_p in rep["methods"].items() if "A_on_B_subset" in by_p.get("A", {})]
    if sub:
        lines.append("Protocol A on the Protocol B subset vs all tasks: " + ", ".join(
            f"{m} {100 * r['A_on_B_subset']:.1f} vs {100 * r['correct']['mean']:.1f}" for m, r in sub))
    rg = rep.get("rep_gap_A") or {}
    if any(v["n"] for v in rg.values()):
        lines.append("Protocol A |rep 1 - rep 2|: " + ", ".join(
            f"{k} {100 * v['mean_abs']:.1f} pp (n={v['n']})" for k, v in rg.items() if v["n"]))
    rv = rep["repeat_variability_A"]
    if rv:
        lines.append(f"Protocol A repeats: {rv['pairs']} pairs, mean |Δ correct| {100 * rv['mean_abs_delta']:.1f} pp "
                     f"(max {100 * rv['max_abs_delta']:.1f}), {100 * rv['mean_task_flip_rate']:.1f}% of tasks flip.")
    if rep["missing_harnesses"]:
        lines.append(f"Missing harnesses: {', '.join(rep['missing_harnesses'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a unified 6-seed re-evaluation")
    ap.add_argument("out", nargs="?", default="runs/v10_unified_6seed")
    ap.add_argument("--ref", default="arch_coev")
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    rep = summarize(Path(a.out), ref=a.ref, n_boot=a.n_boot)
    (Path(a.out) / "unified_summary.json").write_text(json.dumps(rep, indent=2))
    print(render(rep))


if __name__ == "__main__":
    main()

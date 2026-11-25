"""Score saved Protocol A scripts on the blind relocation variants, and summarize (docs/BLIND_PROTOCOL.md).

Every Protocol A eval of the unified re-evaluation saves the one script each harness wrote per task on `low`
(`<eval>.code.jsonl`). This runs those scripts, unchanged and offline, on `data/v2_blind` (8 probe-agnostic
locations per test task) and compares methods by seed-paired blind accuracy: the share of scored blind variants
a script answers correctly, per task, averaged over a method's Protocol A repeats.

    nix-shell --run "python scripts/rebuttal/score_blind.py score"     # -> <out>/evals/*.blind.json
    nix-shell --run "python scripts/rebuttal/score_blind.py summary"   # -> <out>/blind_summary.json

The same two commands with `--blind data/v2_heldout_mech` score and summarize the held-out failure-mechanism
variants of E2 (docs/NEXT_EXPERIMENTS.md) into `*.heldout.json` / `<out>/heldout_summary.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.gsb.harness import matches, run_variant  # noqa: E402
from meta_rsi.eval.unified_summary import BONF_K, hier_bootstrap_ci, holm, sign_flip_p, t_interval  # noqa: E402

_CODE_RE = re.compile(r"^(?P<tag>.+)_A_rep(?P<rep>\d+)\.code\.jsonl$")
BANDS = ("0-30", "30-60", "60-80", "80-89", "large")
LAT_DIST_BINS = ((0.0, 5.0), (5.0, 15.0), (15.0, 90.0))  # |lat - nearest relocating-probe latitude|, degrees
# Addendum 2: the probes' failure mechanisms are hemisphere-symmetric, so also bin by distance in |latitude|
PROBE_ABS_LATS = (58.3, 56.2, 86.0)
DOMAINS = ("equatorial", "utm_valid", "polar")  # |lat| <= 8; inside UTM's 84N / 80S limits; beyond them
# E2 (docs/NEXT_EXPERIMENTS.md): held-out failure mechanisms; one confirmatory comparison on far transfer
# docs/NEXT_EXPERIMENTS.md Amendments 1-2: near = us_ft, albers, nodata_nan; adjacent = csv_lon360; far = raster_southup;
# descriptive = vector_z, multipart (failed the calibration rule)
MECHS = ("us_ft", "albers", "nodata_nan", "csv_lon360", "raster_southup", "vector_z", "multipart")
TRANSFERS = ("near", "adjacent", "far", "descriptive")
E2_CONFIRMATORY = ("arch_coev", "naive", "transfer:far")
# blind area weight: within a band the centres are area-uniform, so the weight of a band is its share of the
# sphere's area (sin hi - sin lo), split over the task's scored variants in that band; `large` tasks are
# area-uniform over one interval and need no reweighting
BAND_AREA = {f"{lo:g}-{hi:g}": math.sin(math.radians(hi)) - math.sin(math.radians(lo))
             for lo, hi in ((0, 30), (30, 60), (60, 80), (80, 89))}


def abs_lat_dist(lat: float) -> float:
    return min(abs(abs(lat) - p) for p in PROBE_ABS_LATS)


def domain(lat: float) -> str:
    if abs(lat) <= 8.0:
        return "equatorial"
    return "polar" if (lat > 84.0 or lat < -80.0) else "utm_valid"


def blind_path(code_file: Path, suffix: str = ".blind.json") -> Path:
    return code_file.with_name(code_file.name.replace(".code.jsonl", suffix))


def load_blind(root: Path) -> dict[str, dict]:
    return {t["task_id"]: t for t in map(json.loads, open(root / "tasks.jsonl"))}


def is_heldout(blind: dict[str, dict]) -> bool:
    """True for the E2 held-out mechanism set (variants carry `mechanism`), False for the blind relocations."""
    return any("mechanism" in v for t in blind.values() for v in t["variants"].values())


def suffix_for(blind: dict[str, dict]) -> str:
    return ".heldout.json" if is_heldout(blind) else ".blind.json"


def cmd_score(a: argparse.Namespace) -> None:
    out, root = Path(a.out), Path(a.blind)
    blind = load_blind(root)
    suffix = getattr(a, "suffix", "") or suffix_for(blind)
    tags = {e["tag"] for e in json.loads((out / "manifest.json").read_text())["entries"]}
    files = [f for f in sorted((out / "evals").glob("*.code.jsonl")) if (m := _CODE_RE.match(f.name)) and m["tag"] in tags]
    # Optional priority ordering (e.g. naive & arch_coev first)
    prio = [p.strip() for p in getattr(a, "prioritize", "").split(",") if p.strip()]
    if prio:
        def _rank(p: Path) -> tuple[int, str]:
            for idx, pref in enumerate(prio):
                if p.name.startswith(pref + "_"):
                    return (idx, p.name)
            return (len(prio), p.name)
        files.sort(key=_rank)
    cache: dict[tuple[str, str], dict] = {}  # (code sha, variant dir) -> run; repeats at temp 0 often share code
    lock = threading.Lock()

    def run(code: str, task: dict, vn: str) -> dict:
        v = task["variants"][vn]
        key = (hashlib.sha256(code.encode()).hexdigest(), v["dir"])
        with lock:
            hit = cache.get(key)
        if hit is None:
            r = run_variant(code, root / v["dir"], a.timeout)
            hit = {"status": r["status"], "value": r.get("value")}
            with lock:
                cache[key] = hit
        ok = hit["status"] == "ok" and matches(hit["value"], v["truth"], task["answer"])
        val = hit["value"]
        return {"status": hit["status"], "value": val if val is None or np.isfinite(val) else str(val), "correct": bool(ok)}

    for f in files:
        dst = blind_path(f, suffix=suffix)
        rows = [json.loads(ln) for ln in open(f)]
        # incremental: keep results already on disk, run only (task, variant) pairs that are missing, e.g. after
        # a mechanism was added to the variant set
        done: dict[str, dict] = {}
        if dst.exists():
            done = {rec["task_id"]: rec["variants"] for rec in json.loads(dst.read_text()).get("records", [])}
        jobs = [(r, vn) for r in rows if r["task_id"] in blind and r.get("code")
                for vn, v in blind[r["task_id"]]["variants"].items() if v["scored"] and vn not in done.get(r["task_id"], {})]
        missing_nocode = any(r["task_id"] in blind and not r.get("code")
                             and {vn for vn, v in blind[r["task_id"]]["variants"].items() if v["scored"]} - set(done.get(r["task_id"], {}))
                             for r in rows)
        if dst.exists() and not jobs and not missing_nocode:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(lambda j: run(j[0]["code"], blind[j[0]["task_id"]], j[1]), jobs))
        by_task: dict[str, dict] = defaultdict(dict)
        for tid, vs in done.items():
            by_task[tid].update(vs)
        for (r, vn), x in zip(jobs, res):
            by_task[r["task_id"]][vn] = x
        recs = []
        for r in rows:
            t = blind.get(r["task_id"])
            if t is None:
                continue
            scored = [vn for vn, v in t["variants"].items() if v["scored"]]
            vs = ({vn: by_task[r["task_id"]][vn] for vn in scored if vn in by_task[r["task_id"]]} if r.get("code")
                  else {vn: {"status": "no_code", "value": None, "correct": False} for vn in scored})
            recs.append({"task_id": r["task_id"], "family": t["family"], "variants": vs})
        n_ok = sum(x["correct"] for rec in recs for x in rec["variants"].values())
        n_all = sum(len(rec["variants"]) for rec in recs)
        dst.write_text(json.dumps({"source": f.name, "blind_root": str(root), "n_tasks": len(recs),
                                   "accuracy": n_ok / max(n_all, 1), "records": recs}))
        print(f"[{suffix.split('.')[1]}] {f.name}: {n_ok}/{n_all} = {n_ok / max(n_all, 1):.3f} ({time.time() - t0:.0f}s, cache {len(cache)})",
              flush=True)


def _lat_bin(d: float) -> str:
    for lo, hi in LAT_DIST_BINS:
        if lo <= d < hi:
            return f"{lo:g}-{hi:g}"
    return f"{LAT_DIST_BINS[-1][0]:g}-{LAT_DIST_BINS[-1][1]:g}"


def summarize(out: Path, root: Path, refs: list[str], n_boot: int = 4000, drop_reused_rep1: bool = False) -> dict:
    blind = load_blind(root)
    manifest = json.loads((out / "manifest.json").read_text())
    tag_info = {e["tag"]: e for e in manifest["entries"]}
    task_ids = sorted(blind)
    tidx = {t: i for i, t in enumerate(task_ids)}
    fam = np.array([blind[t]["family"] for t in task_ids])
    heldout = is_heldout(blind)
    suffix = suffix_for(blind)

    def strata(task: dict, vn: str, low_ok: bool) -> list[str]:
        v = task["variants"][vn]
        if heldout:
            s = ["all", f"mech:{v['mechanism']}", f"transfer:{v['transfer']}"]
            # near and far cover different families (far is mostly the raster families): `matched` keeps only
            # tasks that have both, so near vs far is not a family comparison
            if {"near", "far"} <= {x["transfer"] for x in task["variants"].values() if x["scored"]}:
                s.append(f"matched:{v['transfer']}")
            if low_ok:  # robustness: the same script was correct on `low` in this eval
                s += [f"lowok:transfer:{v['transfer']}", f"lowok:mech:{v['mechanism']}"]
            return s
        s = ["all", f"band:{v['band']}"]
        if not task["large"]:
            s.append(f"probe_lat_dist:{_lat_bin(v['probe_lat_dist'])}")
            s.append(f"abs_lat_dist:{_lat_bin(abs_lat_dist(v['lat']))}")
            s.append(f"domain:{domain(v['lat'])}")
            if v.get("crosses_antimeridian"):
                s.append("domain:antimeridian")
        return s

    # per (method, seed): list over repeats of {stratum: (n_tasks,) share correct, NaN where the task has none}
    per: dict[tuple[str, int], list[dict[str, np.ndarray]]] = defaultdict(list)
    for f in sorted((out / "evals").glob(f"*{suffix}")):
        m = re.match(r"^(?P<tag>.+)_A_rep(?P<rep>\d+)" + re.escape(suffix) + "$", f.name)
        if not m or m["tag"] not in tag_info:
            continue
        e = tag_info[m["tag"]]
        if drop_reused_rep1 and e.get("reused_A_rep1") and m["rep"] == "1":
            continue  # pre-registered backend drop rule (decided by unified_summary's backend_check)
        num: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(task_ids)))
        den: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(task_ids)))
        ev = f.with_name(f.name.replace(suffix, ".json"))
        ev_recs = json.loads(ev.read_text()).get("records", []) if ev.exists() else []
        low_ok = {r["task_id"] for r in ev_recs if r.get("variants", {}).get("low", {}).get("correct")}
        for rec in json.loads(f.read_text())["records"]:
            t = blind[rec["task_id"]]
            for vn, x in rec["variants"].items():
                i = tidx[rec["task_id"]]
                for s in strata(t, vn, rec["task_id"] in low_ok):
                    num[s][i] += x["correct"]
                    den[s][i] += 1
                if not heldout:  # area weight: undoes the equal quota per band (Addendum 3)
                    band = t["variants"][vn]["band"]
                    n_b = sum(1 for wn in rec["variants"] if t["variants"][wn]["band"] == band)
                    w = (BAND_AREA[band] if band in BAND_AREA else 1.0) / n_b
                    num["all_areaw"][i] += w * x["correct"]
                    den["all_areaw"][i] += w
        if heldout and ev_recs:  # `low` correct, per task with a scored variant: the base for `bothlow` (Amendment 2)
            for rec in json.loads(f.read_text())["records"]:
                if rec["variants"]:
                    num["low"][tidx[rec["task_id"]]] += float(rec["task_id"] in low_ok)
                    den["low"][tidx[rec["task_id"]]] += 1
        # the same scripts on the designed non-low variants (high / polar / antimeridian / mid / zone_edge)
        if ev_recs:
            for rec in ev_recs:
                if rec["task_id"] not in tidx:
                    continue
                for vn, x in rec["variants"].items():
                    if vn != "low":
                        num["designed"][tidx[rec["task_id"]]] += x["correct"]
                        den["designed"][tidx[rec["task_id"]]] += 1
        per[(e["method"], e["seed"])].append({s: np.where(den[s] > 0, num[s] / np.where(den[s] > 0, den[s], 1.0), np.nan) for s in num})

    methods = list(dict.fromkeys(e["method"] for e in manifest["entries"]))
    all_strata = sorted({s for reps in per.values() for r in reps for s in r})

    def seed_matrix(method: str, s: str) -> dict[int, np.ndarray]:
        out_ = {}
        for (mm, seed), reps in per.items():
            if mm == method and any(s in r for r in reps):
                with warnings.catch_warnings():  # all-NaN columns (no variant of the task in s) stay NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out_[seed] = np.nanmean(np.stack([r[s] for r in reps if s in r]), axis=0)
        return out_

    def paired(mth: str, ref: str, s: str):
        """Seed-paired per-seed mean differences on the tasks both methods have in stratum s, every seed."""
        sa, sb = seed_matrix(mth, s), seed_matrix(ref, s)
        ss = sorted(set(sa) & set(sb))
        if not ss:
            return None
        x1, x2 = np.stack([sa[k] for k in ss]), np.stack([sb[k] for k in ss])
        ok = ~np.isnan(x1).any(0) & ~np.isnan(x2).any(0)
        if not ok.any():
            return None
        return ss, x1[:, ok], x2[:, ok], fam[ok], [float(np.mean(x1[i, ok] - x2[i, ok])) for i in range(len(ss))]

    def paired_bothlow(mth: str, ref: str, s: str):
        """Retention difference without baseline bias: per seed, only tasks both methods solved on `low` in every
        repeat of that seed; the task set may differ between seeds (Amendment 2)."""
        sa, sb, la, lb = seed_matrix(mth, s), seed_matrix(ref, s), seed_matrix(mth, "low"), seed_matrix(ref, "low")
        ss = sorted(set(sa) & set(sb) & set(la) & set(lb))
        diffs, ns = [], []
        for k in ss:
            ok = (~np.isnan(sa[k]) & ~np.isnan(sb[k]) & (np.nan_to_num(la[k]) == 1.0) & (np.nan_to_num(lb[k]) == 1.0))
            if not ok.any():
                return None
            diffs.append(float(np.mean(sa[k][ok] - sb[k][ok])))
            ns.append(int(ok.sum()))
        return (ss, diffs, ns) if ss else None

    rep = {"blind_root": str(root), "kind": "heldout" if heldout else "blind", "drop_reused_rep1": drop_reused_rep1,
           "methods": {}, "vs_ref": {}}
    for mth in methods:
        rep["methods"][mth] = {}
        for s in all_strata:
            sm = seed_matrix(mth, s)
            vals = [float(np.nanmean(x)) for x in sm.values() if not np.isnan(x).all()]  # lowok can be empty
            if vals:
                rep["methods"][mth][s] = {"seeds": sorted(sm), "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    for ref in refs:
        rows = []
        for mth in methods:
            if mth == ref:
                continue
            pr = paired(mth, ref, "all")
            if pr is None:
                continue
            seeds, x1, x2, fm, diffs = pr
            row = {"method": mth, "seeds": seeds, "diff": float(np.mean(diffs)), "per_seed_diff": diffs,
                   "sign_flip_p": sign_flip_p(diffs),
                   "t_ci95": t_interval(diffs), "t_ci_bonf": t_interval(diffs, 1 - 0.05 / BONF_K),
                   "boot_ci95": hier_bootstrap_ci(x1, x2, fm, n_boot=n_boot)}
            # where the difference lives: per stratum, seed-paired mean difference + 95% hierarchical bootstrap CI
            row["by_stratum"] = {}
            row["by_stratum_ci95"] = {}
            row["by_stratum_p"] = {}
            row["by_stratum_t95"] = {}
            for s in all_strata:
                if s.startswith("lowok:") or s == "low":
                    continue  # method-specific task sets: in the method table only; `bothlow` is the paired version
                pr = paired(mth, ref, s)
                if pr is None:
                    continue
                _, s_x1, s_x2, s_fm, s_diffs = pr
                row["by_stratum"][s] = float(np.mean(s_diffs))
                row["by_stratum_p"][s] = sign_flip_p(s_diffs)
                row["by_stratum_t95"][s] = t_interval(s_diffs)
                row["by_stratum_ci95"][s] = hier_bootstrap_ci(s_x1, s_x2, s_fm, n_boot=n_boot)
            if heldout:
                row["bothlow"] = {}
                for s in [f"transfer:{t}" for t in TRANSFERS] + [f"mech:{m}" for m in MECHS]:
                    pb = paired_bothlow(mth, ref, s)
                    if pb is not None:
                        ss, d, ns = pb
                        row["bothlow"][s] = {"diff": float(np.mean(d)), "per_seed_diff": d, "n_tasks_per_seed": ns,
                                             "sign_flip_p": sign_flip_p(d), "t_ci95": t_interval(d)}
            rows.append(row)
        for r, p in zip(rows, holm([r["sign_flip_p"] for r in rows])):
            r["holm_p"] = p
        rep["vs_ref"][ref] = rows
    if heldout:  # E2's one confirmatory comparison: exact sign-flip over seeds, established only if p < 0.05
        mth, ref, s = E2_CONFIRMATORY
        pr = paired(mth, ref, s)
        if pr is not None:
            seeds, _, _, _, diffs = pr
            p = sign_flip_p(diffs)
            c = rep["confirmatory"] = {"method": mth, "ref": ref, "stratum": s, "seeds": seeds, "diff": float(np.mean(diffs)),
                                       "per_seed_diff": diffs, "sign_flip_p": p, "t_ci95": t_interval(diffs),
                                       "established": bool(p < 0.05)}
            pb = paired_bothlow(mth, ref, s)
            c["bothlow_diff"] = float(np.mean(pb[1])) if pb else None
            ret = [rep["methods"].get(x, {}).get(f"lowok:{s}", {}).get("mean") for x in (mth, ref)]
            c["lowok"] = ret
            # Amendment 2 readings: robustness transfer needs a same-sign `bothlow` difference; a mechanism that
            # both methods survive (>= 90%) or fail (<= 10%) on solved tasks cannot separate them
            c["uninformative"] = bool(None not in ret and (min(ret) >= 0.9 or max(ret) <= 0.1))
            c["robustness_transfer"] = bool(c["established"] and not c["uninformative"] and c["bothlow_diff"] is not None
                                            and np.sign(c["bothlow_diff"]) == np.sign(c["diff"]))
    return rep


def render_heldout(rep: dict) -> str:
    cols = (["designed", "all"] + [f"transfer:{t}" for t in TRANSFERS] + [f"mech:{m}" for m in MECHS]
            + ["matched:near", "matched:far", "low", "lowok:transfer:near", "lowok:transfer:far"])
    labels = [c.split(":")[-1] if c.split(":")[0] in ("transfer", "mech") or ":" not in c
              else f"{c.split(':')[0][:5]}:{c.split(':')[-1]}" for c in cols]
    lines = ["Held-out mechanism accuracy (E2; share of scored variants correct; mean over seeds). `near`: us_ft, albers,",
             "nodata_nan; `adjacent`: csv_lon360; `far`: raster_southup. `match`: only tasks with both near and far variants.",
             "`lowok`: only tasks the same script got right on `low` (task sets differ by method, so descriptive only).",
             "`low`: the same scripts on `low`. Comparisons add `bothlow`: per seed, only tasks both methods solved on `low`.",
             "Drop reused rep 1: " + str(rep.get("drop_reused_rep1")),
             f"{'method':16s} " + " ".join(f"{lab:>11s}" for lab in labels)]
    for m, d in rep["methods"].items():
        lines.append(f"{m:16s} " + " ".join(f"{100 * d[c]['mean']:10.1f}%" if c in d else f"{'-':>11s}" for c in cols))
    if "confirmatory" in rep:
        c = rep["confirmatory"]
        lo, hi = c["t_ci95"]
        lines.append(f"\nCONFIRMATORY (E2): {c['method']} - {c['ref']} on {c['stratum']}: {100 * c['diff']:+.1f} pp, "
                     f"per seed [{', '.join(f'{100 * x:+.1f}' for x in c['per_seed_diff'])}], exact sign-flip "
                     f"p={c['sign_flip_p']:.4f}, t5 95% [{100 * lo:+.1f}, {100 * hi:+.1f}], n_seeds={len(c['seeds'])} -> "
                     + ("ESTABLISHED" if c["established"] else "not established"))
        bl = "-" if c.get("bothlow_diff") is None else f"{100 * c['bothlow_diff']:+.1f} pp"
        rt = ", ".join("-" if x is None else f"{100 * x:.1f}%" for x in c.get("lowok", []))
        lines.append(f"  bothlow {bl}; lowok (method, ref) {rt}; uninformative={c.get('uninformative')}; "
                     f"robustness transfer={c.get('robustness_transfer')}")
    for ref, rows in rep["vs_ref"].items():
        lines.append(f"\nvs {ref} (descriptive): seed-paired difference (pp) [t5 95%], sign-flip p")
        for r in rows:
            parts = []
            for key in ("all", "transfer:near", "transfer:adjacent", "transfer:far", "matched:near", "matched:far"):
                if key in r["by_stratum"] or key == "all":
                    d = r["diff"] if key == "all" else r["by_stratum"][key]
                    lo, hi = r["t_ci95"] if key == "all" else r["by_stratum_t95"][key]
                    pv = r["sign_flip_p"] if key == "all" else r["by_stratum_p"][key]
                    parts.append(f"{key.split(':')[0][:5] + ':' if key.startswith('matched') else ''}{key.split(':')[-1]} "
                                 f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] p={pv:.3f}")
            for key in ("transfer:near", "transfer:far"):
                if key in r.get("bothlow", {}):
                    b = r["bothlow"][key]
                    parts.append(f"bothlow:{key.split(':')[1]} {100 * b['diff']:+.1f} p={b['sign_flip_p']:.3f}")
            lines.append(f"  {r['method']:16s} " + "  | ".join(parts) + f"  n_seeds={len(r['seeds'])}")
    return "\n".join(lines)


def render(rep: dict) -> str:
    if rep.get("kind") == "heldout":
        return render_heldout(rep)
    bins = [f"{lo:g}-{hi:g}" for lo, hi in LAT_DIST_BINS]
    cols = (["designed", "all", "all_areaw"] + [f"band:{b}" for b in BANDS] + [f"probe_lat_dist:{b}" for b in bins]
            + [f"abs_lat_dist:{b}" for b in bins] + [f"domain:{d}" for d in DOMAINS])
    lines = ["Blind-variant accuracy (share of scored blind variants correct; mean over seeds). `designed`: the same",
             "scripts on the designed non-low variants; `all`: every scored blind variant; `all_areaw`: each band weighted",
             "by its share of the sphere's area; then by band, by signed (`sig`) and |lat| (`abs`) distance to the nearest relocating-probe",
             "latitude, and by mechanism domain (local families). Drop reused rep 1: " + str(rep.get("drop_reused_rep1")),
             f"{'method':16s} " + " ".join(f"{(c.split(':')[0][:3] + ':' if 'dist' in c else '') + c.split(':')[-1]:>11s}" for c in cols)]
    for m, d in rep["methods"].items():
        lines.append(f"{m:16s} " + " ".join(f"{100 * d[c]['mean']:10.1f}%" if c in d else f"{'-':>11s}" for c in cols))
    for ref, rows in rep["vs_ref"].items():
        lines.append(f"\nvs {ref} (descriptive): seed-paired difference (pp), sign-flip p, Holm p, t5 95%, "
                     f"t5 Bonferroni/{BONF_K}, bootstrap seeds x families")
        for r in rows:
            lo, hi = r["boot_ci95"]
            (tl, th), (bl, bh) = r["t_ci95"], r["t_ci_bonf"]
            far_str = ""
            for key, lab in (("probe_lat_dist:15-90", ">=15deg"), ("abs_lat_dist:15-90", "|lat|>=15deg")):
                if key in r.get("by_stratum_t95", {}):
                    flo, fhi = r["by_stratum_t95"][key]
                    far_str += f"  | {lab}: {100 * r['by_stratum'][key]:+.1f} [{100 * flo:+.1f}, {100 * fhi:+.1f}]"
            lines.append(f"  {r['method']:16s} {100 * r['diff']:+6.1f}  p={r['sign_flip_p']:.3f}  holm={r['holm_p']:.3f}  "
                         f"t[{100 * tl:+.1f}, {100 * th:+.1f}]  bonf[{100 * bl:+.1f}, {100 * bh:+.1f}]  "
                         f"boot[{100 * lo:+.1f}, {100 * hi:+.1f}]  n_seeds={len(r['seeds'])}{far_str}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs/v10_unified_6seed"))
    ap.add_argument("--blind", default=str(REPO / "data/v2_blind"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--workers", type=int, default=18)
    s.add_argument("--timeout", type=float, default=25.0)
    s.add_argument("--suffix", default="", help="Output suffix (default: .heldout.json for the E2 set, else .blind.json)")
    s.add_argument("--prioritize", default="", help="Comma-separated method prefixes to score first (e.g. naive,arch_coev)")
    m = sub.add_parser("summary")
    m.add_argument("--refs", default="naive,arch_coev")
    m.add_argument("--n-boot", type=int, default=4000)
    m.add_argument("--drop-reused-rep1", choices=("auto", "yes", "no"), default="auto",
                   help="auto: follow backend_check in <out>/unified_summary.json (run unified_summary first)")
    a = ap.parse_args()
    if a.cmd == "score":
        cmd_score(a)
    else:
        drop = a.drop_reused_rep1 == "yes"
        if a.drop_reused_rep1 == "auto":
            us = Path(a.out) / "unified_summary.json"
            if not us.exists():
                raise SystemExit(f"{us} missing: run `python -m meta_rsi.eval.unified_summary {a.out}` first, or pass --drop-reused-rep1")
            drop = bool(json.loads(us.read_text())["backend_check"]["applied"])
        rep = summarize(Path(a.out), Path(a.blind), a.refs.split(","), a.n_boot, drop_reused_rep1=drop)
        (Path(a.out) / f"{rep['kind']}_summary.json").write_text(json.dumps(rep, indent=2))
        print(render(rep))


if __name__ == "__main__":
    main()

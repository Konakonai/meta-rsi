"""Score saved Protocol A scripts on the blind relocation variants, and summarize (docs/BLIND_PROTOCOL.md).

Every Protocol A eval of the unified re-evaluation saves the one script each harness wrote per task on `low`
(`<eval>.code.jsonl`). This runs those scripts, unchanged and offline, on `data/v2_blind` (8 probe-agnostic
locations per test task) and compares methods by seed-paired blind accuracy: the share of scored blind variants
a script answers correctly, per task, averaged over a method's Protocol A repeats.

    nix-shell --run "python scripts/rebuttal/score_blind.py score"     # -> <out>/evals/*.blind.json
    nix-shell --run "python scripts/rebuttal/score_blind.py summary"   # -> <out>/blind_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.gsb.harness import matches, run_variant  # noqa: E402
from meta_rsi.eval.unified_summary import hier_bootstrap_ci, holm, sign_flip_p  # noqa: E402

_CODE_RE = re.compile(r"^(?P<tag>.+)_A_rep(?P<rep>\d+)\.code\.jsonl$")
BANDS = ("0-30", "30-60", "60-80", "80-89", "large")
LAT_DIST_BINS = ((0.0, 5.0), (5.0, 15.0), (15.0, 90.0))  # |lat - nearest relocating-probe latitude|, degrees


def blind_path(code_file: Path) -> Path:
    return code_file.with_name(code_file.name.replace(".code.jsonl", ".blind.json"))


def load_blind(root: Path) -> dict[str, dict]:
    return {t["task_id"]: t for t in map(json.loads, open(root / "tasks.jsonl"))}


def cmd_score(a: argparse.Namespace) -> None:
    out, root = Path(a.out), Path(a.blind)
    blind = load_blind(root)
    tags = {e["tag"] for e in json.loads((out / "manifest.json").read_text())["entries"]}
    files = [f for f in sorted((out / "evals").glob("*.code.jsonl")) if (m := _CODE_RE.match(f.name)) and m["tag"] in tags]
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
        dst = blind_path(f)
        rows = [json.loads(ln) for ln in open(f)]
        if dst.exists() and len(json.loads(dst.read_text()).get("records", [])) == len(rows):
            continue
        t0 = time.time()
        jobs = [(r, vn) for r in rows if r["task_id"] in blind and r.get("code")
                for vn, v in blind[r["task_id"]]["variants"].items() if v["scored"]]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(lambda j: run(j[0]["code"], blind[j[0]["task_id"]], j[1]), jobs))
        by_task: dict[str, dict] = defaultdict(dict)
        for (r, vn), x in zip(jobs, res):
            by_task[r["task_id"]][vn] = x
        recs = []
        for r in rows:
            t = blind.get(r["task_id"])
            if t is None:
                continue
            scored = [vn for vn, v in t["variants"].items() if v["scored"]]
            vs = by_task.get(r["task_id"]) or {vn: {"status": "no_code", "value": None, "correct": False} for vn in scored}
            recs.append({"task_id": r["task_id"], "family": t["family"], "variants": vs})
        n_ok = sum(x["correct"] for rec in recs for x in rec["variants"].values())
        n_all = sum(len(rec["variants"]) for rec in recs)
        dst.write_text(json.dumps({"source": f.name, "blind_root": str(root), "n_tasks": len(recs),
                                   "accuracy": n_ok / max(n_all, 1), "records": recs}))
        print(f"[blind] {f.name}: {n_ok}/{n_all} = {n_ok / max(n_all, 1):.3f} ({time.time() - t0:.0f}s, cache {len(cache)})",
              flush=True)


def _lat_bin(d: float) -> str:
    for lo, hi in LAT_DIST_BINS:
        if lo <= d < hi:
            return f"{lo:g}-{hi:g}"
    return f"{LAT_DIST_BINS[-1][0]:g}-{LAT_DIST_BINS[-1][1]:g}"


def summarize(out: Path, root: Path, refs: list[str], n_boot: int = 4000) -> dict:
    blind = load_blind(root)
    manifest = json.loads((out / "manifest.json").read_text())
    tag_info = {e["tag"]: e for e in manifest["entries"]}
    task_ids = sorted(blind)
    tidx = {t: i for i, t in enumerate(task_ids)}
    fam = np.array([blind[t]["family"] for t in task_ids])

    def strata(task: dict, vn: str) -> list[str]:
        v = task["variants"][vn]
        s = ["all", f"band:{v['band']}"]
        if not task["large"]:
            s.append(f"probe_lat_dist:{_lat_bin(v['probe_lat_dist'])}")
        return s

    # per (method, seed): list over repeats of {stratum: (n_tasks,) share correct, NaN where the task has none}
    per: dict[tuple[str, int], list[dict[str, np.ndarray]]] = defaultdict(list)
    for f in sorted((out / "evals").glob("*.blind.json")):
        m = re.match(r"^(?P<tag>.+)_A_rep(?P<rep>\d+)\.blind\.json$", f.name)
        if not m or m["tag"] not in tag_info:
            continue
        e = tag_info[m["tag"]]
        num: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(task_ids)))
        den: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(task_ids)))
        for rec in json.loads(f.read_text())["records"]:
            t = blind[rec["task_id"]]
            for vn, x in rec["variants"].items():
                for s in strata(t, vn):
                    num[s][tidx[rec["task_id"]]] += x["correct"]
                    den[s][tidx[rec["task_id"]]] += 1
        # the same scripts on the designed non-low variants (high / polar / antimeridian / mid / zone_edge)
        ev = f.with_name(f.name.replace(".blind.json", ".json"))
        if ev.exists():
            for rec in json.loads(ev.read_text()).get("records", []):
                if rec["task_id"] not in tidx:
                    continue
                for vn, x in rec["variants"].items():
                    if vn != "low":
                        num["designed"][tidx[rec["task_id"]]] += x["correct"]
                        den["designed"][tidx[rec["task_id"]]] += 1
        per[(e["method"], e["seed"])].append({s: np.where(den[s] > 0, num[s] / np.maximum(den[s], 1), np.nan) for s in num})

    methods = list(dict.fromkeys(e["method"] for e in manifest["entries"]))
    all_strata = sorted({s for reps in per.values() for r in reps for s in r})

    def seed_matrix(method: str, s: str) -> dict[int, np.ndarray]:
        return {seed: np.nanmean(np.stack([r[s] for r in reps if s in r]), axis=0)
                for (mm, seed), reps in per.items() if mm == method and any(s in r for r in reps)}

    rep = {"blind_root": str(root), "methods": {}, "vs_ref": {}}
    for mth in methods:
        rep["methods"][mth] = {}
        for s in all_strata:
            sm = seed_matrix(mth, s)
            if sm:
                vals = [float(np.nanmean(x)) for x in sm.values()]
                rep["methods"][mth][s] = {"seeds": sorted(sm), "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    for ref in refs:
        rows = []
        for mth in methods:
            if mth == ref:
                continue
            a, b = seed_matrix(mth, "all"), seed_matrix(ref, "all")
            seeds = sorted(set(a) & set(b))
            if not seeds:
                continue
            x1, x2 = np.stack([a[s] for s in seeds]), np.stack([b[s] for s in seeds])
            ok = ~np.isnan(x1).any(0) & ~np.isnan(x2).any(0)
            diffs = [float(np.mean(x1[i, ok] - x2[i, ok])) for i in range(len(seeds))]
            row = {"method": mth, "seeds": seeds, "diff": float(np.mean(diffs)), "per_seed_diff": diffs,
                   "sign_flip_p": sign_flip_p(diffs),
                   "boot_ci95": hier_bootstrap_ci(x1[:, ok], x2[:, ok], fam[ok], n_boot=n_boot)}
            # where the difference lives: per stratum, seed-paired mean difference
            row["by_stratum"] = {}
            for s in all_strata:
                sa, sb = seed_matrix(mth, s), seed_matrix(ref, s)
                ss = sorted(set(sa) & set(sb))
                if ss:
                    row["by_stratum"][s] = float(np.mean([np.nanmean(sa[k] - sb[k]) for k in ss]))
            rows.append(row)
        for r, p in zip(rows, holm([r["sign_flip_p"] for r in rows])):
            r["holm_p"] = p
        rep["vs_ref"][ref] = rows
    return rep


def render(rep: dict) -> str:
    cols = ["designed", "all"] + [f"band:{b}" for b in BANDS] + [f"probe_lat_dist:{lo:g}-{hi:g}" for lo, hi in LAT_DIST_BINS]
    lines = ["Blind-variant accuracy (share of scored blind variants correct; mean over seeds). `designed`: the same",
             "scripts on the designed non-low variants; `all`: every scored blind variant, then by band and by",
             "|latitude - nearest relocating-probe latitude| (local families)",
             f"{'method':16s} " + " ".join(f"{c.split(':')[-1]:>9s}" for c in cols)]
    for m, d in rep["methods"].items():
        lines.append(f"{m:16s} " + " ".join(f"{100 * d[c]['mean']:8.1f}%" if c in d else f"{'-':>9s}" for c in cols))
    for ref, rows in rep["vs_ref"].items():
        lines.append(f"\nvs {ref}: seed-paired difference (pp), sign-flip p, Holm p, 95% hierarchical bootstrap")
        for r in rows:
            lo, hi = r["boot_ci95"]
            lines.append(f"  {r['method']:16s} {100 * r['diff']:+6.1f}  p={r['sign_flip_p']:.3f}  holm={r['holm_p']:.3f}  "
                         f"[{100 * lo:+.1f}, {100 * hi:+.1f}]  n_seeds={len(r['seeds'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs/v10_unified_6seed"))
    ap.add_argument("--blind", default=str(REPO / "data/v2_blind"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--workers", type=int, default=18)
    s.add_argument("--timeout", type=float, default=25.0)
    m = sub.add_parser("summary")
    m.add_argument("--refs", default="naive,arch_coev")
    m.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    if a.cmd == "score":
        cmd_score(a)
    else:
        rep = summarize(Path(a.out), Path(a.blind), a.refs.split(","), a.n_boot)
        (Path(a.out) / "blind_summary.json").write_text(json.dumps(rep, indent=2))
        print(render(rep))


if __name__ == "__main__":
    main()

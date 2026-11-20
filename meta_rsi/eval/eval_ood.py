"""Out-of-distribution (OOD) Polar Stress-Test Evaluator for evolved harnesses.

Tests whether an evolved harness generalizes to polar latitudes (>84N / <80S)
where UTM is undefined (`estimate_utm_crs()` raises RuntimeError) and which are
completely disjoint from the metamorphic gate's in-loop targets (58.3N / -56.2S).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.gsb.families import BY_NAME
from benchmarks.gsb.harness import matches

from ..llm import LLMClient
from ..metamorphic import perturb_workspace
from ..sandbox import load_harness_module, parse_final_number, run_harness_on_workspace, run_python_sandbox

POLAR_TARGETS = (
    ("polar_north", dict(target_lat=86.5, target_lon=25.0, shuffle_rows=True, flip_gpkg_crs=False, seed=101)),
    ("polar_south", dict(target_lat=-85.5, target_lon=-40.0, shuffle_rows=True, flip_gpkg_crs=True, seed=202)),
)


def score_polar_ood(
    task: dict,
    code: str | None,
    data_root: Path,
    *,
    timeout: float = 25.0,
    track_caught: bool = False,
    python: str = sys.executable,
) -> dict:
    """Run `code` on original `low` plus polar North & South OOD relocations."""
    if code is None:
        return {"category": "no_code", "variants": {}, "swallowed": False}

    low_info = task["variants"]["low"]
    low_dir = data_root / low_info["dir"]
    truth = low_info["truth"]
    ans = task["answer"]

    variants: dict[str, dict] = {}
    any_swallowed = False

    def _run_dir(d: Path) -> dict:
        nonlocal any_swallowed
        r = run_python_sandbox(code, d, timeout=timeout, python=python, track_caught=track_caught)
        if track_caught and r.get("caught_exceptions"):
            any_swallowed = True
        if r["timed_out"]:
            return {"status": "timeout"}
        if r["returncode"] != 0:
            tail = (r["stderr"].strip().splitlines() or [""])[-1][:200]
            return {"status": "exception", "stderr": tail}
        val = parse_final_number(r["stdout"])
        if val is None:
            return {"status": "no_number"}
        return {"status": "ok", "value": val, "correct": bool(matches(val, truth, ans))}

    variants["low"] = _run_dir(low_dir)
    for name, kwargs in POLAR_TARGETS:
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / name
            perturb_workspace(low_dir, pdir, **kwargs)
            variants[name] = _run_dir(pdir)

    if any(v["status"] != "ok" for v in variants.values()):
        cat = "error"
    else:
        n_ok = sum(1 for v in variants.values() if v["correct"])
        cat = "correct" if n_ok == len(variants) else ("silent_wrong" if n_ok == 0 else "lucky")
    return {"category": cat, "variants": variants, "swallowed": any_swallowed}


def main() -> None:
    ap = argparse.ArgumentParser(description="Polar OOD (>84N / <80S) stress test for an evolved harness")
    ap.add_argument("--harness", required=True, help="Path to harness.py")
    ap.add_argument("--data-root", default="data/v1")
    ap.add_argument("--out", required=True, help="Output JSON summary path")
    ap.add_argument("--task-backend", default="openai-compat", choices=["openai-compat", "gemini-cli", "gemini-rest"])
    ap.add_argument("--task-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--task-url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--task-temp", type=float, default=0.0)
    ap.add_argument("--task-max-tokens", type=int, default=2048)
    ap.add_argument("--task-no-think", action="store_true")
    ap.add_argument("--max-llm-calls", type=int, default=4)
    ap.add_argument("--exec-timeout", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seen-per-family", type=int, default=5)
    ap.add_argument("--heldout-per-family", type=int, default=20)
    ap.add_argument("--protocol", default="A", choices=["A", "B"], help="A = generate code on equatorial low_dir and replay on polar; B = deploy harness.solve() directly on polar workspaces")
    ap.add_argument("--track-caught", action="store_true")
    a = ap.parse_args()

    data_root = Path(a.data_root)
    all_rows = [json.loads(ln) for ln in open(data_root / "tasks.jsonl")]
    train_fams = {t["family"] for t in all_rows if t["split"] == "train"}

    by_fam_t: dict[str, list[dict]] = defaultdict(list)
    for t in all_rows:
        if t["split"] == "test" and not BY_NAME[t["family"]].large:
            by_fam_t[t["family"]].append(t)

    test_tasks = []
    for fam in sorted(by_fam_t):
        cap = a.seen_per_family if fam in train_fams else a.heldout_per_family
        test_tasks.extend(by_fam_t[fam][:cap] if cap > 0 else by_fam_t[fam])

    llm = LLMClient(
        a.task_backend,
        a.task_model,
        base_url=a.task_url,
        temperature=a.task_temp,
        max_tokens=a.task_max_tokens,
        enable_thinking=not a.task_no_think,
    )
    harness_mod = load_harness_module(Path(a.harness))

    def llm_fn(system: str, user: str) -> str:
        r = llm.complete(system, user)
        if r.raw_error and not r.text:
            raise RuntimeError(f"LLM backend error: {r.raw_error}")
        return r.text

    def eval_one(task: dict) -> dict:
        low_info = task["variants"]["low"]
        low_dir = data_root / low_info["dir"]
        if a.protocol == "A":
            run = run_harness_on_workspace(
                harness_mod,
                task["prompt"],
                low_dir,
                llm_fn,
                max_llm_calls=a.max_llm_calls,
                exec_timeout=a.exec_timeout,
            )
            sc = score_polar_ood(task, run["code"], data_root, timeout=a.exec_timeout, track_caught=a.track_caught)
            return {"family": task["family"], "llm_calls": run["llm_calls"], **sc}

        # Protocol B: deploy harness.solve() directly on each polar workspace (tests live execute-inspect-repair on polar data)
        truth = low_info["truth"]
        ans = task["answer"]
        variants: dict[str, dict] = {}
        any_swallowed = False
        total_calls = 0
        for name, kwargs in POLAR_TARGETS:
            with tempfile.TemporaryDirectory() as tmp:
                pdir = Path(tmp) / name
                perturb_workspace(low_dir, pdir, **kwargs)
                run = run_harness_on_workspace(
                    harness_mod,
                    task["prompt"],
                    pdir,
                    llm_fn,
                    max_llm_calls=a.max_llm_calls,
                    exec_timeout=a.exec_timeout,
                )
                total_calls += run["llm_calls"]
                code = run["code"]
                if code is None:
                    variants[name] = {"status": "no_code"}
                    continue
                r = run_python_sandbox(code, pdir, timeout=a.exec_timeout, track_caught=a.track_caught)
                if a.track_caught and r.get("caught_exceptions"):
                    any_swallowed = True
                if r["timed_out"]:
                    variants[name] = {"status": "timeout"}
                elif r["returncode"] != 0:
                    tail = (r["stderr"].strip().splitlines() or [""])[-1][:200]
                    variants[name] = {"status": "exception", "stderr": tail}
                else:
                    val = parse_final_number(r["stdout"])
                    if val is None:
                        variants[name] = {"status": "no_number"}
                    else:
                        variants[name] = {"status": "ok", "value": val, "correct": bool(matches(val, truth, ans))}
        if any(v["status"] != "ok" for v in variants.values()):
            cat = "error"
        else:
            n_ok = sum(1 for v in variants.values() if v["correct"])
            cat = "correct" if n_ok == len(variants) else ("silent_wrong" if n_ok == 0 else "lucky")
        return {"family": task["family"], "llm_calls": total_calls / max(len(POLAR_TARGETS), 1), "category": cat, "variants": variants, "swallowed": any_swallowed}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        records = list(ex.map(eval_one, test_tasks))
    elapsed = round(time.time() - t0, 1)

    by_fam: dict[str, Counter] = defaultdict(Counter)
    seen_cats, ho_cats = Counter(), Counter()
    swallowed_n = 0
    swallowed_runnable_n = 0
    for r in records:
        c = r["category"]
        by_fam[r["family"]][c] += 1
        if r["swallowed"]:
            swallowed_n += 1
            if c in ("correct", "lucky", "silent_wrong"):
                swallowed_runnable_n += 1
        if r["family"] in train_fams:
            seen_cats[c] += 1
        else:
            ho_cats[c] += 1

    def rates(cnt: Counter) -> dict:
        n = max(sum(cnt.values()), 1)
        return {
            "n": sum(cnt.values()),
            "correct": round(cnt["correct"] / n, 4),
            "lucky": round(cnt["lucky"] / n, 4),
            "silent_wrong": round(cnt["silent_wrong"] / n, 4),
            "error": round((cnt["error"] + cnt["no_code"]) / n, 4),
        }

    all_cnt = seen_cats + ho_cats
    out_obj = {
        "harness": a.harness,
        "protocol": a.protocol,
        "ood_targets": [name for name, _ in POLAR_TARGETS],
        "model": a.task_model,
        "temp": a.task_temp,
        "no_think": a.task_no_think,
        "elapsed_s": elapsed,
        "avg_llm_calls": round(sum(r["llm_calls"] for r in records) / max(len(records), 1), 3),
        "swallowed_count": swallowed_n if a.track_caught else None,
        "swallowed_runnable_count": swallowed_runnable_n if a.track_caught else None,
        "usage": llm.usage.to_dict(),
        "ALL_LOCAL": rates(all_cnt),
        "SEEN_LOCAL_16": rates(seen_cats),
        "HELDOUT_3": rates(ho_cats),
        "per_family": {fam: rates(cnt) for fam, cnt in sorted(by_fam.items())},
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"=== Polar OOD Eval (n={out_obj['ALL_LOCAL']['n']}): {a.harness} ({elapsed}s) ===")
    for k in ("ALL_LOCAL", "SEEN_LOCAL_16", "HELDOUT_3"):
        r = out_obj[k]
        print(
            f"  {k:<14} (n={r['n']:3d}): correct={r['correct']*100:5.1f}%  "
            f"lucky={r['lucky']*100:5.1f}%  silent_wrong={r['silent_wrong']*100:5.1f}%  "
            f"error={r['error']*100:5.1f}%"
        )


if __name__ == "__main__":
    main()

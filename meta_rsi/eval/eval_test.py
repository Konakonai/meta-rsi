"""Evaluate any saved harness.py on the FULL 480-task data/v1 test split (apples-to-apples with zero_shot_v1)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.gsb.harness import matches

from ..evaluator import evaluate_harness
from ..llm import LLMClient
from ..sandbox import load_harness_module, parse_final_number, run_harness_on_workspace, run_python_sandbox


def _evaluate_protocol_b(
    harness_path: Path,
    tasks: list[dict],
    data_root: Path,
    llm: LLMClient,
    *,
    max_llm_calls: int = 4,
    exec_timeout: float = 25.0,
    workers: int = 16,
) -> tuple[list[dict], float]:
    """Protocol B: deploy harness.solve() directly on EVERY variant workspace (live coordinates visible)."""
    harness_mod = load_harness_module(harness_path)

    def llm_fn(system: str, user: str) -> str:
        r = llm.complete(system, user)
        if r.raw_error and not r.text:
            raise RuntimeError(f"LLM backend error: {r.raw_error}")
        return r.text

    def eval_one(task: dict) -> dict:
        var_results: dict[str, dict] = {}
        total_calls = 0
        for vname, vinfo in task["variants"].items():
            vdir = data_root / vinfo["dir"]
            run = run_harness_on_workspace(
                harness_mod,
                task["prompt"],
                vdir,
                llm_fn,
                max_llm_calls=max_llm_calls,
                exec_timeout=exec_timeout,
                python=sys.executable,
            )
            calls = int(run["llm_calls"])
            total_calls += calls
            vr = var_results[vname] = {"status": "ok", "correct": False, "llm_calls": calls}
            code = run["code"]
            if run["harness_error"]:
                vr.update(status="harness_error", detail=run["harness_error"])
                continue
            if not code:
                vr["status"] = "no_code"
                continue
            r = run_python_sandbox(code, vdir, timeout=exec_timeout, python=sys.executable)
            if r["timed_out"]:
                vr["status"] = "timeout"
                continue
            if r["returncode"] != 0:
                vr.update(status="exception", detail=(r["stderr"].strip().splitlines() or [""])[-1])
                continue
            val = parse_final_number(r["stdout"])
            if val is None:
                vr.update(status="no_number", detail=r["stdout"][-200:])
                continue
            vr.update(value=val, correct=bool(matches(val, vinfo["truth"], task["answer"])))

        n_vars = max(len(var_results), 1)
        if any(v["status"] != "ok" for v in var_results.values()):
            cat = "error"
        else:
            n_ok = sum(1 for v in var_results.values() if v["correct"])
            if n_ok == n_vars:
                cat = "correct"
            elif n_ok == 0:
                cat = "silent_wrong"
            else:
                cat = "lucky"
        return {
            "task_id": task["task_id"],
            "family": task["family"],
            "oracle_category": cat,
            "llm_calls": total_calls / n_vars,
            "variant_results": var_results,
        }

    if workers <= 1:
        records = [eval_one(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(eval_one, tasks))
    avg_calls = round(sum(r["llm_calls"] for r in records) / max(len(records), 1), 3)
    return records, avg_calls


def task_records(records: list[dict], tasks: list[dict], protocol: str) -> list[dict]:
    """Compact per-task records for the output JSON (no code or prompts).

    Every variant keeps its raw value next to its truth, so a run can be re-scored offline (e.g. after a
    tolerance change) and paired task by task with another harness or model. Protocol A variants are the
    oracle runs of the one script written on the low variant; Protocol B variants are separate live
    deployments, each with its own LLM calls.
    """
    by_id = {t["task_id"]: t for t in tasks}
    out = []
    for r in records:
        task = by_id[r["task_id"]]
        runs = (r.get("variant_results") if protocol == "B" else r.get("oracle_variants")) or {}
        variants = {}
        for vn, vinfo in task["variants"].items():
            vr = runs.get(vn) or {"status": "no_code"}  # Protocol A without a script runs no variant
            v = vr.get("value")
            entry = {
                "status": vr["status"],
                "value": v if v is None or math.isfinite(v) else str(v),  # keep the file strict JSON
                "truth": vinfo["truth"],
                "correct": bool(vr.get("correct", False)),
            }
            detail = vr.get("detail") or vr.get("stderr") or vr.get("stdout")
            if detail:
                entry["detail"] = str(detail)[-300:]
            if "llm_calls" in vr:
                entry["llm_calls"] = vr["llm_calls"]
            variants[vn] = entry
        rec = {
            "task_id": r["task_id"],
            "family": r["family"],
            "answer": task["answer"],
            "category": r["oracle_category"] or "error",
            "llm_calls": r["llm_calls"],
            "variants": variants,
        }
        if r.get("harness_error"):
            rec["harness_error"] = str(r["harness_error"])[-300:]
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Full test-set evaluation for an evolved harness")
    ap.add_argument("--harness", required=True, help="Path to harness.py")
    ap.add_argument("--data-root", default="data/v1")
    ap.add_argument("--out", required=True, help="Output JSON summary path")
    ap.add_argument("--protocol", default="A", choices=["A", "B"], help="A = script-reuse from low variant; B = live per-variant deployment")
    ap.add_argument("--task-backend", default="openai-compat", choices=["openai-compat", "gemini-cli", "gemini-rest"])
    ap.add_argument("--task-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--task-url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--task-temp", type=float, default=0.0)
    ap.add_argument("--task-max-tokens", type=int, default=2048)
    ap.add_argument("--task-no-think", action="store_true")
    ap.add_argument("--max-llm-calls", type=int, default=4)
    ap.add_argument("--exec-timeout", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seen-per-family", type=int, default=0, help="0 = all test instances per seen family")
    ap.add_argument("--heldout-per-family", type=int, default=0, help="0 = all test instances per heldout family")
    a = ap.parse_args()

    data_root = Path(a.data_root)
    all_rows = [json.loads(ln) for ln in open(data_root / "tasks.jsonl")]
    train_fams = {t["family"] for t in all_rows if t["split"] == "train"}
    if a.seen_per_family or a.heldout_per_family:
        by_fam_t: dict[str, list[dict]] = defaultdict(list)
        for t in all_rows:
            if t["split"] == "test":
                by_fam_t[t["family"]].append(t)
        test_tasks = []
        for fam in sorted(by_fam_t):
            cap = a.seen_per_family if fam in train_fams else a.heldout_per_family
            test_tasks.extend(by_fam_t[fam][:cap] if cap > 0 else by_fam_t[fam])
    else:
        test_tasks = [t for t in all_rows if t["split"] == "test"]

    llm = LLMClient(
        a.task_backend,
        a.task_model,
        base_url=a.task_url,
        api_key=a.api_key or None,
        temperature=a.task_temp,
        max_tokens=a.task_max_tokens,
        enable_thinking=not a.task_no_think,
    )
    t0 = time.time()
    if a.protocol == "B":
        records, avg_calls = _evaluate_protocol_b(
            Path(a.harness),
            test_tasks,
            data_root,
            llm,
            max_llm_calls=a.max_llm_calls,
            exec_timeout=a.exec_timeout,
            workers=a.workers,
        )
    else:
        res = evaluate_harness(
            Path(a.harness),
            test_tasks,
            data_root,
            llm,
            check_metamorphic=False,
            compute_oracle=True,
            skip_dev_exec=True,
            keep_oracle_variants=True,
            max_llm_calls=a.max_llm_calls,
            exec_timeout=a.exec_timeout,
            workers=a.workers,
        )
        records = res["records"]
        avg_calls = res["summary"]["avg_llm_calls"]
    elapsed = round(time.time() - t0, 1)
    task_recs = task_records(records, test_tasks, a.protocol)

    by_fam: dict[str, Counter] = defaultdict(Counter)
    by_var: dict[str, Counter] = defaultdict(Counter)
    seen_cats, ho_cats = Counter(), Counter()
    for r in task_recs:
        c = r["category"]
        by_fam[r["family"]][c] += 1
        if r["family"] in train_fams:
            seen_cats[c] += 1
        else:
            ho_cats[c] += 1
        for vn, vr in r["variants"].items():
            if vr["status"] != "ok":
                by_var[vn]["error"] += 1
            elif vr["correct"]:
                by_var[vn]["correct"] += 1
            else:
                by_var[vn]["wrong"] += 1

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
        "model": a.task_model,
        "temp": a.task_temp,
        "no_think": a.task_no_think,
        "elapsed_s": elapsed,
        "avg_llm_calls": avg_calls,
        "usage": llm.usage.to_dict(),
        "ALL": rates(all_cnt),
        "SEEN_18": rates(seen_cats),
        "HELDOUT_3": rates(ho_cats),
        "per_family": {fam: rates(cnt) for fam, cnt in sorted(by_fam.items())},
    }
    if by_var:
        out_obj["per_variant"] = {
            vn: {
                "n": sum(cnt.values()),
                "correct": round(cnt["correct"] / max(sum(cnt.values()), 1), 4),
                "wrong": round(cnt["wrong"] / max(sum(cnt.values()), 1), 4),
                "error": round(cnt["error"] / max(sum(cnt.values()), 1), 4),
            }
            for vn, cnt in sorted(by_var.items())
        }
    out_obj["records"] = task_recs
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))

    print(f"=== Unseen Test Eval [Protocol {a.protocol}] (n={out_obj['ALL']['n']}): {a.harness} ({elapsed}s, avg_calls={out_obj['avg_llm_calls']:.2f}) ===")
    for k in ("ALL", "SEEN_18", "HELDOUT_3"):
        r = out_obj[k]
        print(
            f"  {k:<10} (n={r['n']:3d}): correct={r['correct']*100:5.1f}%  "
            f"lucky={r['lucky']*100:5.1f}%  silent_wrong={r['silent_wrong']*100:5.1f}%  "
            f"error={r['error']*100:5.1f}%"
        )
    if by_var:
        print("  Per-variant live accuracy:" if a.protocol == "B" else "  Per-variant accuracy of the low-variant script:")
        for vn, vd in out_obj["per_variant"].items():
            print(f"    {vn:<14} (n={vd['n']:3d}): correct={vd['correct']*100:5.1f}%  wrong={vd['wrong']*100:5.1f}%  err={vd['error']*100:5.1f}%")


if __name__ == "__main__":
    main()

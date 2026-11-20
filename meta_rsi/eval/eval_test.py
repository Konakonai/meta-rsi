"""Evaluate any saved harness.py on the FULL 480-task data/v1 test split (apples-to-apples with zero_shot_v1)."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..evaluator import evaluate_harness
from ..llm import LLMClient


def main() -> None:
    ap = argparse.ArgumentParser(description="Full 480-task test-set evaluation for an evolved harness")
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
        temperature=a.task_temp,
        max_tokens=a.task_max_tokens,
        enable_thinking=not a.task_no_think,
    )
    t0 = time.time()
    res = evaluate_harness(
        Path(a.harness),
        test_tasks,
        data_root,
        llm,
        check_metamorphic=False,
        compute_oracle=True,
        skip_dev_exec=True,
        max_llm_calls=a.max_llm_calls,
        exec_timeout=a.exec_timeout,
        workers=a.workers,
    )
    elapsed = round(time.time() - t0, 1)

    by_fam: dict[str, Counter] = defaultdict(Counter)
    seen_cats, ho_cats = Counter(), Counter()
    for r in res["records"]:
        c = r["oracle_category"] or "error"
        by_fam[r["family"]][c] += 1
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
        "model": a.task_model,
        "temp": a.task_temp,
        "no_think": a.task_no_think,
        "elapsed_s": elapsed,
        "avg_llm_calls": res["summary"]["avg_llm_calls"],
        "usage": llm.usage.to_dict(),
        "ALL": rates(all_cnt),
        "SEEN_18": rates(seen_cats),
        "HELDOUT_3": rates(ho_cats),
        "per_family": {fam: rates(cnt) for fam, cnt in sorted(by_fam.items())},
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))

    print(f"=== Unseen Test Eval (n={out_obj['ALL']['n']}): {a.harness} ({elapsed}s, avg_calls={out_obj['avg_llm_calls']:.2f}) ===")
    for k in ("ALL", "SEEN_18", "HELDOUT_3"):
        r = out_obj[k]
        print(
            f"  {k:<10} (n={r['n']:3d}): correct={r['correct']*100:5.1f}%  "
            f"lucky={r['lucky']*100:5.1f}%  silent_wrong={r['silent_wrong']*100:5.1f}%  "
            f"error={r['error']*100:5.1f}%"
        )


if __name__ == "__main__":
    main()

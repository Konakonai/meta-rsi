"""Zero-shot evaluation: sample N answers per task with vLLM, score each on the three variants.

    python scripts/zero_shot.py --model google/gemma-4-E4B-it --root data/v1 --split test --n 8 \
        --think 1 --out results/e4b_test.jsonl

Sampling uses the model's own generation defaults (temperature, top_p, top_k); only n and
max_tokens are set here. `--hint 1` appends an accuracy requirement to every prompt (the
prompt-only baseline); `--hint 2` appends the same requirement plus explicit CRS and geodesic
rules (does the model lack the rules, or fail to apply them). `--completions file.jsonl` re-scores saved answers without a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsb.harness import extract_code, score  # noqa: E402

HINT = "\nDistances, lengths and areas must be accurate to within 1%, wherever the data is on the Earth."
RULES = """
Rules for coordinate systems:
- Check the CRS of every input layer and bring all layers to one CRS before combining them.
- Do not measure distances, lengths or areas in degrees (EPSG:4326) or in Web Mercator (EPSG:3857).
- If the data spans less than 6 degrees of longitude, project every layer to the UTM CRS returned by `gdf.estimate_utm_crs()` and measure there.
- If the data spans more than 6 degrees, measure on the ellipsoid with `pyproj.Geod(ellps="WGS84")` (`inv`, `geometry_length`, `geometry_area_perimeter`), or use an equal-area projection for areas."""
HINTS = {0: "", 1: HINT, 2: HINT + RULES}
CATS = ["correct", "lucky", "silent_wrong", "error", "no_code", "no_answer"]


def generate(tasks, a):
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"enable_thinking": bool(a.think)}
    if a.effort:
        kw["reasoning_effort"] = a.effort
    prompts = [tok.apply_chat_template([{"role": "user", "content": t["prompt"] + HINTS[a.hint]}],
                                       tokenize=False, add_generation_prompt=True, **kw) for t in tasks]
    llm = LLM(model=a.model, max_model_len=a.max_tokens + 1024, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_seqs)  # hybrid models (Qwen3.8 DeltaNet) need one state block per sequence
    sp = llm.get_default_sampling_params()
    sp.n, sp.max_tokens = a.n, a.max_tokens
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0
    rows = []
    for t, o in zip(tasks, outs):
        for i, c in enumerate(o.outputs):
            rows.append({"task_id": t["task_id"], "sample": i, "text": c.text, "finish_reason": c.finish_reason,
                         "n_tokens": len(c.token_ids)})
    meta = {"model": a.model, "think": a.think, "effort": a.effort, "hint": a.hint, "n": a.n,
            "max_tokens": a.max_tokens, "sampling": repr(sp), "generate_seconds": gen_s}
    return rows, meta


def score_rows(rows, tasks_by_id, root, a):
    def one(r):
        t = tasks_by_id[r["task_id"]]
        if a.think and "qwen" in a.model.lower() and "</think>" not in r["text"]:
            return {"category": "no_answer", "reward": 0.0, "variants": {}}  # cut off while still thinking
        return score(t, extract_code(r["text"]), root, timeout=a.timeout)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, rows))
    for r, s in zip(rows, results):
        r.update(category=s["category"], variants=s["variants"], family=tasks_by_id[r["task_id"]]["family"])
    return time.time() - t0


def summarize(rows):
    fam = defaultdict(lambda: {"cats": Counter(), "tasks": defaultdict(list)})
    for r in rows:
        f = fam[r["family"]]
        f["cats"][r["category"]] += 1
        f["tasks"][r["task_id"]].append(r["category"] == "correct")
    out = {}
    total = {"cats": Counter(), "tasks": {}}
    for name, f in fam.items():
        n = sum(f["cats"].values())
        out[name] = {"samples": n, **{c: f["cats"][c] / n for c in CATS},
                     "pass@1": f["cats"]["correct"] / n,
                     "pass@n": sum(any(v) for v in f["tasks"].values()) / len(f["tasks"])}
        total["cats"].update(f["cats"])
        total["tasks"].update(f["tasks"])
    n = sum(total["cats"].values())
    out["ALL"] = {"samples": n, **{c: total["cats"][c] / n for c in CATS}, "pass@1": total["cats"]["correct"] / n,
                  "pass@n": sum(any(v) for v in total["tasks"].values()) / len(total["tasks"])}
    return out


def print_table(summary):
    cols = ["samples", "pass@1", "pass@n"] + CATS
    print("family".ljust(26) + "".join(c.rjust(13) for c in cols))
    for name, s in sorted(summary.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
        print(name.ljust(26) + "".join((f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c])).rjust(13)
                                       for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--root", default="data/v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--think", type=int, default=1)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--hint", type=int, default=0, choices=sorted(HINTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--completions", default=None, help="re-score saved rows instead of generating")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.root)
    tasks = [t for t in map(json.loads, open(root / "tasks.jsonl")) if t["split"] == a.split]
    if a.limit:
        tasks = tasks[:a.limit]
    by_id = {t["task_id"]: t for t in tasks}
    if a.completions:
        rows = [r for r in map(json.loads, open(a.completions)) if r["task_id"] in by_id]
        meta = {"rescored_from": a.completions}
    else:
        rows, meta = generate(tasks, a)
    meta["score_seconds"] = score_rows(rows, by_id, root, a)
    summary = summarize(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"meta": meta, "summary": summary}, open(out.with_suffix(".summary.json"), "w"), indent=1)
    print(json.dumps(meta))
    print_table(summary)


if __name__ == "__main__":
    main()

"""Collect verified single-turn distillation trajectories from an evolved harness.

Strictly operates on a disjoint `train` slice (e.g. indices [20:55] per seen family, disjoint from
both evolution dev sets [0:19] and the 480 unseen test tasks). Selection uses ONLY the weak `low`
variant (in `weak` mode) or `low` + label-free metamorphic invariance (in `metamorphic` mode) —
never the 3-location test oracle.

Outputs JSONL records formatted for bare `seed.py` prompts (`SYSTEM_PROMPT` + raw `task_prompt`),
so downstream SFT/GRPO distills multi-turn harness knowledge into single-turn base weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.gsb.harness import matches

from ..evaluator import _run_metamorphic_check
from ..harnesses.seed import SYSTEM_PROMPT as SEED_SYSTEM_PROMPT
from ..llm import LLMClient
from ..sandbox import load_harness_module, parse_final_number, run_harness_on_workspace, run_python_sandbox
from ..verifier_agent import load_probe_fn


def load_probes_from_dir(probe_dir: Path | None, cache_dir: Path) -> list[tuple[str, object]] | None:
    """Load certified `.py` probe modules from `probe_dir`, if provided."""
    if probe_dir is None or not probe_dir.exists():
        return None
    probes: list[tuple[str, object]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pfile in sorted(probe_dir.glob("*.py")):
        pfn = load_probe_fn(pfile.read_text(), cache_dir, name=pfile.stem)
        probes.append((pfile.stem, pfn))
    return probes or None


def select_train_slice(data_root: Path, *, train_offset: int = 20, train_per_family: int = 35) -> list[dict]:
    rows = [json.loads(ln) for ln in open(data_root / "tasks.jsonl")]
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for t in rows:
        if t["split"] == "train":
            by_fam[t["family"]].append(t)
    selected: list[dict] = []
    for fam in sorted(by_fam):
        selected.extend(by_fam[fam][train_offset : train_offset + train_per_family])
    return selected


def collect_trajectories(
    *,
    harness_path: Path,
    data_root: Path,
    out_jsonl: Path,
    llm: LLMClient,
    filter_mode: str = "metamorphic",
    custom_probes: list[tuple[str, object]] | None = None,
    train_offset: int = 20,
    train_per_family: int = 35,
    max_llm_calls: int = 4,
    max_exec_calls: int = 12,
    exec_timeout: float = 20.0,
    workers: int = 16,
) -> dict:
    if filter_mode not in {"weak", "metamorphic"}:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    mod = load_harness_module(harness_path)
    tasks = select_train_slice(data_root, train_offset=train_offset, train_per_family=train_per_family)
    t0 = time.time()

    def _one(task: dict) -> dict:
        low_info = task["variants"]["low"]
        low_dir = data_root / low_info["dir"]
        h = run_harness_on_workspace(
            mod,
            task["prompt"],
            low_dir,
            lambda sys_p, usr_p: llm.complete(sys_p, usr_p).text,
            max_llm_calls=max_llm_calls,
            max_exec_calls=max_exec_calls,
            exec_timeout=exec_timeout,
        )
        code = h.get("code")
        rec = {
            "task_id": task["task_id"],
            "family": task["family"],
            "turns": h.get("llm_calls", 0),
            "low_ok": False,
            "meta_ok": None,
            "kept": False,
            "uses_3857": bool(code and "3857" in code),
            "code": code,
            "prompt": task["prompt"],
        }
        if not code:
            return rec
        r = run_python_sandbox(code, low_dir, timeout=exec_timeout)
        if r["timed_out"] or r["returncode"] != 0:
            return rec
        val = parse_final_number(r["stdout"])
        if val is None:
            return rec
        low_ok = bool(matches(val, low_info["truth"], task["answer"]))
        rec["low_ok"] = low_ok
        if not low_ok:
            return rec

        if filter_mode == "weak":
            rec["kept"] = True
            return rec

        m_ok, _, _ = _run_metamorphic_check(
            code,
            low_dir,
            val,
            task["answer"],
            timeout=exec_timeout,
            include_domain_hint=False,
            custom_probes=custom_probes,
        )
        rec["meta_ok"] = bool(m_ok)
        rec["kept"] = bool(m_ok)
        return rec

    results: list[dict] = []
    if workers <= 1:
        for t in tasks:
            results.append(_one(t))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_one, tasks):
                results.append(r)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept_rows = []
    per_fam_kept: Counter = Counter()
    per_fam_total: Counter = Counter()
    for r in results:
        per_fam_total[r["family"]] += 1
        if r["kept"] and r["code"]:
            per_fam_kept[r["family"]] += 1
            kept_rows.append(
                {
                    "task_id": r["task_id"],
                    "family": r["family"],
                    "system": SEED_SYSTEM_PROMPT,
                    "user": r["prompt"],
                    "assistant": f"```python\n{r['code'].strip()}\n```",
                    "turns": r["turns"],
                    "low_ok": r["low_ok"],
                    "meta_ok": r["meta_ok"],
                    "uses_3857": r["uses_3857"],
                }
            )

    with open(out_jsonl, "w") as f:
        for row in kept_rows:
            f.write(json.dumps(row) + "\n")

    n_total = len(results)
    n_low = sum(1 for r in results if r["low_ok"])
    n_kept = len(kept_rows)
    n_blocked = sum(1 for r in results if r["low_ok"] and r["meta_ok"] is False)
    n_kept_3857 = sum(1 for row in kept_rows if row["uses_3857"])
    summary = {
        "harness": str(harness_path),
        "filter_mode": filter_mode,
        "train_offset": train_offset,
        "train_per_family": train_per_family,
        "n_total": n_total,
        "n_low_ok": n_low,
        "n_kept": n_kept,
        "n_meta_blocked": n_blocked,
        "kept_rate": round(n_kept / max(n_total, 1), 4),
        "kept_3857_count": n_kept_3857,
        "kept_3857_rate": round(n_kept_3857 / max(n_kept, 1), 4),
        "avg_llm_calls": round(sum(r["turns"] for r in results) / max(n_total, 1), 3),
        "per_family_kept": {f: per_fam_kept[f] for f in sorted(per_fam_total)},
        "elapsed_s": round(time.time() - t0, 1),
        "usage": llm.usage.to_dict(),
    }
    summary_path = out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect distillation trajectories from an evolved harness")
    ap.add_argument("--harness", required=True)
    ap.add_argument("--data-root", default="data/v1")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--filter-mode", default="metamorphic", choices=["weak", "metamorphic"])
    ap.add_argument("--probe-dir", default="", help="Optional dir of certified metamorphic probe .py files")
    ap.add_argument("--train-offset", type=int, default=20)
    ap.add_argument("--train-per-family", type=int, default=35)
    ap.add_argument("--task-backend", default="openai-compat", choices=["openai-compat", "mock"])
    ap.add_argument("--task-model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--task-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--task-temp", type=float, default=0.0)
    ap.add_argument("--task-max-tokens", type=int, default=2048)
    ap.add_argument("--task-no-think", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    llm = LLMClient(
        a.task_backend,
        a.task_model,
        base_url=a.task_url,
        temperature=a.task_temp,
        max_tokens=a.task_max_tokens,
        enable_thinking=not a.task_no_think,
    )
    probes = (
        load_probes_from_dir(Path(a.probe_dir), Path(a.out).parent / "_probe_mods")
        if a.probe_dir
        else None
    )
    summary = collect_trajectories(
        harness_path=Path(a.harness),
        data_root=Path(a.data_root),
        out_jsonl=Path(a.out),
        llm=llm,
        filter_mode=a.filter_mode,
        custom_probes=probes,
        train_offset=a.train_offset,
        train_per_family=a.train_per_family,
        workers=a.workers,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["usage"]["errors"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()

"""Main meta-rsi Harness Evolution CLI.

Example (local smoke test with Gemini CLI):
    python -m meta_rsi.evolve --exp smoke_meta --gate metamorphic --generations 3 \
        --dev-per-family 1 --test-seen-per-family 1 --test-heldout-per-family 2 \
        --task-backend gemini-cli --task-model gemini-3.5-flash-lite \
        --meta-backend gemini-cli --meta-model gemini-4-hydrogen

Example (on Colab G4 with local vLLM serving Gemma 4 E4B):
    python -m meta_rsi.evolve --exp g4_e4b_meta --gate metamorphic --generations 6 \
        --task-backend openai-compat --task-url http://127.0.0.1:8000/v1 --task-model google/gemma-4-E4B-it \
        --meta-backend gemini-rest --meta-model gemini-4-argon
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import shutil
import time
from pathlib import Path

from .archive import ArchiveEntry, LineageArchive, evaluate_gate
from .evaluator import evaluate_harness, load_task_splits
from .llm import LLMClient
from .meta_agent import propose_mutation


def _fmt_pct(v: float | None) -> str:
    return "   -  " if v is None else f"{v * 100:5.1f}%"


def _print_row(
    gen: int,
    parent: int | None,
    accepted: bool,
    dev_low: float,
    in_loop: float,
    ora_dev: float | None,
    ora_seen: float | None,
    ora_ho: float | None,
    err: float,
    lucky: float | None,
    calls: float,
    note: str,
) -> None:
    p_str = "seed" if parent is None else f"g{parent:02d}"
    st = "ACCEPT" if accepted else "REJECT"
    print(
        f"g{gen:02d}  {p_str:>4}  {st:<6}  "
        f"{_fmt_pct(dev_low)}  {_fmt_pct(in_loop)}  "
        f"{_fmt_pct(ora_dev)}  {_fmt_pct(ora_seen)}  {_fmt_pct(ora_ho)}  "
        f"{_fmt_pct(err)}  {_fmt_pct(lucky)}  {calls:4.2f}  {note}",
        flush=True,
    )


def _save_eval(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1))


def run_evolution(
    *,
    exp_dir: Path,
    data_root: Path,
    gate_mode: str,
    generations: int,
    task_llm: LLMClient,
    meta_llm: LLMClient,
    dev_per_family: int = 1,
    dev_offset: int = 0,
    test_seen_per_family: int = 1,
    test_heldout_per_family: int = 6,
    max_llm_calls: int = 4,
    max_regress: int = 0,
    allow_stepping_stone: bool = True,
    exec_timeout: float = 25.0,
    workers: int = 4,
    seed: int = 0,
    seed_harness_path: Path | None = None,
    eval_oracle_all: bool = False,
    limit_dev: int = 0,
    include_domain_hint: bool = False,
    track_caught: bool = False,
) -> LineageArchive:
    exp_dir.mkdir(parents=True, exist_ok=True)
    splits = load_task_splits(
        data_root,
        dev_per_family=dev_per_family,
        dev_offset=dev_offset,
        test_seen_per_family=test_seen_per_family,
        test_heldout_per_family=test_heldout_per_family,
    )
    dev_tasks = splits["dev"][:limit_dev] if limit_dev else splits["dev"]
    test_seen = splits["test_seen"]
    test_heldout = splits["test_heldout"]
    check_meta = gate_mode == "metamorphic"
    use_oracle_gate = gate_mode == "oracle"
    rng = random.Random(seed)

    config = {
        "gate_mode": gate_mode,
        "generations": generations,
        "dev_tasks": len(dev_tasks),
        "test_seen": len(test_seen),
        "test_heldout": len(test_heldout),
        "task_backend": task_llm.backend,
        "task_model": task_llm.model,
        "meta_backend": meta_llm.backend,
        "meta_model": meta_llm.model,
        "max_llm_calls": max_llm_calls,
        "max_regress": max_regress,
        "include_domain_hint": include_domain_hint,
        "track_caught": track_caught,
        "seed": seed,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    header = (
        "Gen  From  Status  DevLow  InLoop  OraDev  OraSn   OraHO   DevErr  Lucky  Calls  Note\n"
        + "-" * 108
    )
    print(f"=== meta-rsi Harness Evolution | exp={exp_dir.name} | gate={gate_mode} | "
          f"dev={len(dev_tasks)}, test_seen={len(test_seen)}, heldout={len(test_heldout)} ===", flush=True)
    print(header, flush=True)

    archive = LineageArchive(exp_dir, gate_mode)
    eval_cache: dict[int, dict] = {}
    history_path = exp_dir / "history.jsonl"

    # --- Generation 0: Seed Harness ---
    g0_dir = exp_dir / "gen_000"
    g0_dir.mkdir(parents=True, exist_ok=True)
    src_seed = seed_harness_path or (Path(__file__).resolve().parent / "harnesses" / "seed.py")
    g0_harness = g0_dir / "harness.py"
    shutil.copy2(src_seed, g0_harness)

    t0 = time.time()
    g0_dev = evaluate_harness(
        g0_harness,
        dev_tasks,
        data_root,
        task_llm,
        check_metamorphic=check_meta,
        include_domain_hint=include_domain_hint,
        use_oracle_in_loop=use_oracle_gate,
        track_caught=track_caught,
        compute_oracle=True,
        max_llm_calls=max_llm_calls,
        exec_timeout=exec_timeout,
        workers=workers,
    )
    _save_eval(g0_dir / "dev_eval.json", g0_dev)

    g0_seen = evaluate_harness(
        g0_harness,
        test_seen,
        data_root,
        task_llm,
        check_metamorphic=False,
        compute_oracle=True,
        max_llm_calls=max_llm_calls,
        exec_timeout=exec_timeout,
        workers=workers,
    ) if test_seen else None
    g0_ho = evaluate_harness(
        g0_harness,
        test_heldout,
        data_root,
        task_llm,
        check_metamorphic=False,
        compute_oracle=True,
        max_llm_calls=max_llm_calls,
        exec_timeout=exec_timeout,
        workers=workers,
    ) if test_heldout else None
    if g0_seen:
        _save_eval(g0_dir / "test_seen_eval.json", g0_seen)
    if g0_ho:
        _save_eval(g0_dir / "test_heldout_eval.json", g0_ho)

    s0 = g0_dev["summary"]
    e0 = ArchiveEntry(
        gen=0,
        parent_gen=None,
        harness_path=str(g0_harness),
        rationale="Gen-0 seed baseline harness",
        dev_low_rate=s0["dev_low_rate"],
        in_loop_rate=s0["in_loop_rate"],
        dev_error_rate=s0["dev_error_rate"],
        avg_llm_calls=s0["avg_llm_calls"],
        solved_ids=s0["solved_ids_in_loop"],
        oracle_dev_rate=s0.get("oracle_correct_rate"),
        oracle_dev_lucky=s0.get("oracle_lucky_rate"),
        oracle_test_seen_rate=g0_seen["summary"].get("oracle_correct_rate") if g0_seen else None,
        oracle_test_heldout_rate=g0_ho["summary"].get("oracle_correct_rate") if g0_ho else None,
    )
    archive.add(e0)
    eval_cache[0] = g0_dev
    archive.save()

    _print_row(
        0,
        None,
        True,
        e0.dev_low_rate,
        e0.in_loop_rate,
        e0.oracle_dev_rate,
        e0.oracle_test_seen_rate,
        e0.oracle_test_heldout_rate,
        e0.dev_error_rate,
        e0.oracle_dev_lucky,
        e0.avg_llm_calls,
        f"seed ({time.time() - t0:.1f}s)",
    )
    with open(history_path, "a") as fh:
        fh.write(json.dumps({
            "gen": 0,
            "parent_gen": None,
            "accepted": True,
            "gate_reason": "seed",
            "rationale": e0.rationale,
            "dev_low_rate": e0.dev_low_rate,
            "in_loop_rate": e0.in_loop_rate,
            "dev_error_rate": e0.dev_error_rate,
            "avg_llm_calls": e0.avg_llm_calls,
            "oracle_dev_rate": e0.oracle_dev_rate,
            "oracle_dev_lucky": e0.oracle_dev_lucky,
            "oracle_test_seen_rate": e0.oracle_test_seen_rate,
            "oracle_test_heldout_rate": e0.oracle_test_heldout_rate,
            "task_usage": task_llm.usage.to_dict(),
            "meta_usage": meta_llm.usage.to_dict(),
        }) + "\n")

    # --- Evolution Loop ---
    for gen in range(1, generations + 1):
        tg = time.time()
        g_dir = exp_dir / f"gen_{gen:03d}"
        g_dir.mkdir(parents=True, exist_ok=True)
        parent, partner = archive.select_parent(rng)
        parent_eval = eval_cache[parent.gen]

        try:
            rationale, new_code = propose_mutation(
                meta_llm,
                parent,
                parent_eval,
                archive.attempts,
                gate_mode,
                partner=partner,
            )
        except Exception as exc:
            msg = f"meta_propose_failed: {exc}"
            archive.record_attempt({
                "gen": gen,
                "parent_gen": parent.gen,
                "accepted": False,
                "gate_reason": msg,
                "rationale": "",
            })
            archive.save()
            print(f"g{gen:02d}  g{parent.gen:02d}  ERROR   {msg}", flush=True)
            continue

        cand_harness = g_dir / "harness.py"
        cand_harness.write_text(new_code)
        (g_dir / "rationale.txt").write_text(rationale + "\n")
        parent_lines = Path(parent.harness_path).read_text().splitlines(keepends=True)
        cand_lines = new_code.splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(parent_lines, cand_lines, fromfile=f"gen_{parent.gen:03d}", tofile=f"gen_{gen:03d}"))
        (g_dir / "diff.patch").write_text(diff)

        cand_dev = evaluate_harness(
            cand_harness,
            dev_tasks,
            data_root,
            task_llm,
            check_metamorphic=check_meta,
            include_domain_hint=include_domain_hint,
            use_oracle_in_loop=use_oracle_gate,
            track_caught=track_caught,
            compute_oracle=True,
            max_llm_calls=max_llm_calls,
            exec_timeout=exec_timeout,
            workers=workers,
        )
        _save_eval(g_dir / "dev_eval.json", cand_dev)
        decision = evaluate_gate(
            gate_mode,
            parent,
            cand_dev,
            max_regress=max_regress,
            allow_stepping_stone=allow_stepping_stone,
        )

        ora_seen_rate = None
        ora_ho_rate = None
        if decision.accepted or eval_oracle_all:
            if test_seen:
                cand_seen = evaluate_harness(
                    cand_harness,
                    test_seen,
                    data_root,
                    task_llm,
                    check_metamorphic=False,
                    compute_oracle=True,
                    max_llm_calls=max_llm_calls,
                    exec_timeout=exec_timeout,
                    workers=workers,
                )
                _save_eval(g_dir / "test_seen_eval.json", cand_seen)
                ora_seen_rate = cand_seen["summary"].get("oracle_correct_rate")
            if test_heldout:
                cand_ho = evaluate_harness(
                    cand_harness,
                    test_heldout,
                    data_root,
                    task_llm,
                    check_metamorphic=False,
                    compute_oracle=True,
                    max_llm_calls=max_llm_calls,
                    exec_timeout=exec_timeout,
                    workers=workers,
                )
                _save_eval(g_dir / "test_heldout_eval.json", cand_ho)
                ora_ho_rate = cand_ho["summary"].get("oracle_correct_rate")

        sc = cand_dev["summary"]
        if decision.accepted:
            entry = ArchiveEntry(
                gen=gen,
                parent_gen=parent.gen,
                harness_path=str(cand_harness),
                rationale=rationale,
                dev_low_rate=sc["dev_low_rate"],
                in_loop_rate=sc["in_loop_rate"],
                dev_error_rate=sc["dev_error_rate"],
                avg_llm_calls=sc["avg_llm_calls"],
                solved_ids=sc["solved_ids_in_loop"],
                oracle_dev_rate=sc.get("oracle_correct_rate"),
                oracle_dev_lucky=sc.get("oracle_lucky_rate"),
                oracle_test_seen_rate=ora_seen_rate,
                oracle_test_heldout_rate=ora_ho_rate,
                recombined_from=partner.gen if partner else None,
            )
            archive.add(entry)
            eval_cache[gen] = cand_dev

        archive.record_attempt({
            "gen": gen,
            "parent_gen": parent.gen,
            "partner_gen": partner.gen if partner else None,
            "accepted": decision.accepted,
            "gate_reason": decision.reason,
            "newly_solved": decision.newly_solved,
            "regressed": decision.regressed,
            "meta_blocked": decision.meta_blocked,
            "rationale": rationale,
        })
        archive.save()

        _print_row(
            gen,
            parent.gen,
            decision.accepted,
            sc["dev_low_rate"],
            sc["in_loop_rate"],
            sc.get("oracle_correct_rate"),
            ora_seen_rate,
            ora_ho_rate,
            sc["dev_error_rate"],
            sc.get("oracle_lucky_rate"),
            sc["avg_llm_calls"],
            f"{decision.reason} ({time.time() - tg:.1f}s)",
        )
        with open(history_path, "a") as fh:
            fh.write(json.dumps({
                "gen": gen,
                "parent_gen": parent.gen,
                "accepted": decision.accepted,
                "gate_reason": decision.reason,
                "newly_solved": decision.newly_solved,
                "regressed": decision.regressed,
                "meta_blocked": decision.meta_blocked,
                "rationale": rationale,
                "dev_low_rate": sc["dev_low_rate"],
                "in_loop_rate": sc["in_loop_rate"],
                "dev_error_rate": sc["dev_error_rate"],
                "avg_llm_calls": sc["avg_llm_calls"],
                "oracle_dev_rate": sc.get("oracle_correct_rate"),
                "oracle_dev_lucky": sc.get("oracle_lucky_rate"),
                "oracle_test_seen_rate": ora_seen_rate,
                "oracle_test_heldout_rate": ora_ho_rate,
                "task_usage": task_llm.usage.to_dict(),
                "meta_usage": meta_llm.usage.to_dict(),
            }) + "\n")

    best = archive.best()
    print("-" * 108, flush=True)
    print(
        f"Best in archive: gen_{best.gen:03d} | in_loop={best.in_loop_rate:.1%} | "
        f"oracle_dev={_fmt_pct(best.oracle_dev_rate).strip()} | "
        f"oracle_seen={_fmt_pct(best.oracle_test_seen_rate).strip()} | "
        f"oracle_heldout={_fmt_pct(best.oracle_test_heldout_rate).strip()}",
        flush=True,
    )
    print(f"Task LLM usage: {task_llm.usage.to_dict()}", flush=True)
    print(f"Meta LLM usage: {meta_llm.usage.to_dict()}", flush=True)
    return archive


def main() -> None:
    ap = argparse.ArgumentParser(description="meta-rsi: Metamorphic-Gated Recursive Self-Improvement for Agent Harnesses")
    ap.add_argument("--exp", required=True, help="Experiment name under --runs-dir")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data-root", default="data/v1")
    ap.add_argument("--gate", default="metamorphic", choices=["naive", "preserve", "metamorphic", "oracle"])
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--dev-per-family", type=int, default=1)
    ap.add_argument("--dev-offset", type=int, default=0)
    ap.add_argument("--test-seen-per-family", type=int, default=1)
    ap.add_argument("--test-heldout-per-family", type=int, default=6)
    ap.add_argument("--limit-dev", type=int, default=0, help="Optional cap on dev task count for fast smoke runs")
    ap.add_argument("--max-llm-calls", type=int, default=4)
    ap.add_argument("--max-regress", type=int, default=0)
    ap.add_argument("--no-stepping-stone", action="store_true")
    ap.add_argument("--meta-domain-hints", action="store_true", help="Include GIS/tabular root-cause hints in metamorphic diagnostics (off by default for strict domain neutrality)")
    ap.add_argument("--track-caught-exceptions", action="store_true", help="Use sys.monitoring to track swallowed exceptions in dev scripts")
    ap.add_argument("--exec-timeout", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-oracle-all", action="store_true")

    ap.add_argument("--task-backend", default="openai-compat", choices=["openai-compat", "gemini-cli", "gemini-rest", "mock"])
    ap.add_argument("--task-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--task-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--task-temp", type=float, default=0.4)
    ap.add_argument("--task-max-tokens", type=int, default=4096)
    ap.add_argument("--task-no-think", action="store_true")

    ap.add_argument("--meta-backend", default="gemini-cli", choices=["openai-compat", "gemini-cli", "gemini-rest", "mock"])
    ap.add_argument("--meta-model", default="gemini-4-argon")
    ap.add_argument("--meta-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--meta-temp", type=float, default=0.5)
    ap.add_argument("--meta-max-tokens", type=int, default=16384)
    ap.add_argument("--meta-thinking-level", default="low", choices=["minimal", "low", "medium", "high"])
    a = ap.parse_args()

    task_llm = LLMClient(
        a.task_backend,
        a.task_model,
        base_url=a.task_url,
        temperature=a.task_temp,
        max_tokens=a.task_max_tokens,
        enable_thinking=not a.task_no_think,
    )
    meta_llm = LLMClient(
        a.meta_backend,
        a.meta_model,
        base_url=a.meta_url,
        temperature=a.meta_temp,
        max_tokens=a.meta_max_tokens,
        thinking_level=a.meta_thinking_level,
    )
    run_evolution(
        exp_dir=Path(a.runs_dir) / a.exp,
        data_root=Path(a.data_root),
        gate_mode=a.gate,
        generations=a.generations,
        task_llm=task_llm,
        meta_llm=meta_llm,
        dev_per_family=a.dev_per_family,
        dev_offset=a.dev_offset,
        test_seen_per_family=a.test_seen_per_family,
        test_heldout_per_family=a.test_heldout_per_family,
        max_llm_calls=a.max_llm_calls,
        max_regress=a.max_regress,
        allow_stepping_stone=not a.no_stepping_stone,
        exec_timeout=a.exec_timeout,
        workers=a.workers,
        seed=a.seed,
        eval_oracle_all=a.eval_oracle_all,
        limit_dev=a.limit_dev,
        include_domain_hint=a.meta_domain_hints,
        track_caught=a.track_caught_exceptions,
    )


if __name__ == "__main__":
    main()

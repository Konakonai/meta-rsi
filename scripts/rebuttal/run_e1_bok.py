"""E1 Pre-Registered Experiment: Teacher-Free Best-of-K Self-Distillation (docs/NEXT_EXPERIMENTS.md).

Strictly implements every pre-registered requirement of E1 (incl. Amendment 1 & coupled sampling):
  1. Sample K=16 completions per task from bare `google/gemma-4-26B-A4B-it` (`seed.py`, temp=0.8,
     max_tokens=2048, no_think) on disjoint `train[20:60]` (600 tasks).
  2. Score all 600 x 16 = 9,600 candidates incrementally on CPU (append per task, resumable):
       - weak check (`low_ok`, exact 1% oracle tolerance)
       - E1-fixed metamorphic gate (built-in probes + 3 probes from `scripts/rebuttal/probes_e1/`
         with configurable `--rel-tol`, set per Amendment 2 before launching)
       - offline oracle check (ONLY for reporting training-set oracle-correct share; NEVER used for selection)
  3. Define T = tasks with >=1 candidate passing BOTH weak check and E1 gate.
     For 6 replicates (rep 0..5; single coupled draw seed 100..105 per replicate; train seeds 42..47):
       - For each task in T, permute its candidate list ONCE with the replicate's RNG:
           * `bok_weak_r{i}` picks the FIRST candidate in the permutation with `low_ok`
           * `bok_meta_r{i}` picks the FIRST candidate in the permutation with `low_ok` AND `meta_ok`
         -> Both arms have uniform marginals over their eligible pools AND identical task set T,
            while differing ONLY when the weak pick failed the metamorphic gate!
       - Train 12 LoRAs (r=16, alpha=32, 2 ep, lr=1e-4, bs=16, max_seq_len=2560 — exact `sft_meta` recipe).
  4. Evaluate bare base model (`google/gemma-4-26B-A4B-it`) + all 12 LoRAs on the SAME vLLM server
     (bare `seed.py`, temp=0.0, uerr==0, saving per-task `records` + `.code.jsonl`):
       - Protocol A on all 450 GSB v2 test tasks
       - Protocol B on the 100-task unified subset (`--seen-per-family 5 --heldout-per-family 5`)
       - 8-city Global OSM (`data/osm_global`, 96 tasks)

Supports per-stage execution (`--stages sample,score,train,eval,pack`), `--skip-upload`,
`--download-only`, and `--dry-run-cpu`. NEVER launch on `w3-rebuttal` while `unified_6seed_eval` runs!

Usage:
    nix-shell --run "python -u scripts/rebuttal/run_e1_bok.py --dry-run-cpu"
    nix-shell --run "python -u scripts/rebuttal/run_e1_bok.py --session w3-rebuttal --rel-tol 0.01"
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.wave_s4_2.run_single_g4_wave2 import _run_colab, upload_chunked  # noqa: E402

MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_OUT_DIR = REPO / "runs/v12_e1_bok"


def build_matched_arms(
    candidates_by_task: dict[str, list[dict]],
    task_meta: dict[str, dict],
    *,
    system_prompt: str,
    replicates: int = 6,
    draw_seed_base: int = 100,
    out_dir: Path,
) -> dict:
    """Construct coupled `bok_weak_r{i}.jsonl` and `bok_meta_r{i}.jsonl` on identical task set T.

    Coupled single-RNG permutation: for each replicate and each task in T, permute candidates once;
    `bok_weak` takes the first `low_ok` candidate, `bok_meta` takes the first `low_ok & meta_ok`
    candidate. Marginals are uniform within each pool, and the two arms differ iff `bok_weak`'s pick
    failed the metamorphic gate.
    """
    t_ids = sorted(
        tid for tid, cands in candidates_by_task.items()
        if any(c["low_ok"] and c["meta_ok"] for c in cands)
    )
    per_fam_T = Counter(task_meta[tid]["family"] for tid in t_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_stats = []

    for rep in range(replicates):
        draw_seed = draw_seed_base + rep
        rng = random.Random(draw_seed)
        weak_rows, meta_rows = [], []
        n_diff = 0
        weak_ora_ok, meta_ora_ok = 0, 0

        for tid in t_ids:
            perm = list(candidates_by_task[tid])
            rng.shuffle(perm)
            pick_w = next(c for c in perm if c["low_ok"])
            pick_m = next(c for c in perm if c["low_ok"] and c["meta_ok"])
            if pick_w is not pick_m and pick_w["code"].strip() != pick_m["code"].strip():
                n_diff += 1
            weak_ora_ok += int(bool(pick_w.get("oracle_ok")))
            meta_ora_ok += int(bool(pick_m.get("oracle_ok")))
            t = task_meta[tid]
            for dst, pick in ((weak_rows, pick_w), (meta_rows, pick_m)):
                dst.append({
                    "task_id": tid,
                    "family": t["family"],
                    "system": system_prompt,
                    "user": t["prompt"],
                    "assistant": f"```python\n{pick['code'].strip()}\n```",
                    "low_ok": bool(pick["low_ok"]),
                    "meta_ok": bool(pick["meta_ok"]),
                    "oracle_ok": bool(pick.get("oracle_ok")),
                })

        for name, rows in ((f"bok_weak_r{rep}.jsonl", weak_rows), (f"bok_meta_r{rep}.jsonl", meta_rows)):
            with open(out_dir / name, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

        n_t = max(len(t_ids), 1)
        rep_stats.append({
            "replicate": rep,
            "draw_seed": draw_seed,
            "train_seed": 42 + rep,
            "n_tasks_T": len(t_ids),
            "n_diff_scripts": n_diff,
            "frac_diff_scripts": round(n_diff / n_t, 4),
            "weak_oracle_correct_n": weak_ora_ok,
            "weak_oracle_correct_rate": round(weak_ora_ok / n_t, 4),
            "meta_oracle_correct_n": meta_ora_ok,
            "meta_oracle_correct_rate": round(meta_ora_ok / n_t, 4),
        })

    audit = {
        "n_total_tasks": len(candidates_by_task),
        "n_tasks_T": len(t_ids),
        "per_family_T": dict(sorted(per_fam_T.items())),
        "coupled_single_rng": True,
        "replicates": rep_stats,
    }
    (out_dir / "e1_selection_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


REMOTE_E1_DRIVER = r'''
import json, os, random, shutil, subprocess, sys, tarfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

stop_hb = threading.Event()
def _hb():
    t0 = time.time()
    while not stop_hb.wait(20.0):
        print(f"[e1-bok hb] elapsed={time.time()-t0:.0f}s", flush=True)
threading.Thread(target=_hb, daemon=True).start()

def kill_vllm():
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints|VLLM::|EngineCore"], check=False)
    time.sleep(3)

def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def cast_lora_bf16(src_dir: Path, dst_dir: Path) -> Path:
    import torch
    from safetensors.torch import load_file, save_file
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        src = src_dir / fname
        if not src.exists():
            continue
        dst = dst_dir / fname
        if fname == "adapter_model.safetensors":
            tensors = load_file(str(src))
            casted = {k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype != torch.bfloat16 else v) for k, v in tensors.items()}
            save_file(casted, str(dst))
        else:
            shutil.copy2(src, dst)
    return dst_dir

def start_vllm(model: str, log_path: Path, lora_modules: list[tuple[str, Path]] | None = None) -> subprocess.Popen:
    kill_vllm()
    env = os.environ.copy()
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--host", "127.0.0.1", "--port", "8000",
        "--gpu-memory-utilization", "0.86", "--max-model-len", "8192", "--max-num-seqs", "64",
        "--enable-lora", "--max-loras", "8", "--max-lora-rank", "16",
    ]
    if lora_modules:
        cmd.append("--lora-modules")
        for lname, ldir in lora_modules:
            bf = cast_lora_bf16(ldir, Path("/content/e1_loras_bf16") / lname)
            cmd.append(f"{lname}={bf}")
    print(f"[e1-bok] launching vLLM ({log_path.name}, loras={[x[0] for x in (lora_modules or [])]})...", flush=True)
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while not http_ok("http://127.0.0.1:8000/v1/models"):
        if proc.poll() is not None:
            print(log_path.read_text()[-3000:], file=sys.stderr, flush=True)
            raise RuntimeError(f"vLLM exited rc={proc.returncode}")
        if time.time() - t0 > 900:
            raise TimeoutError("Timed out waiting for vLLM")
        time.sleep(5)
    print(f"[e1-bok] vLLM READY in {time.time()-t0:.1f}s", flush=True)
    return proc

def is_clean(p: Path, min_n: int) -> bool:
    try:
        d = json.loads(p.read_text())
        u = d.get("usage") or {}
        n = int((d.get("ALL") or {}).get("n", 0))
        return n >= min_n and int(u.get("errors", 1)) == 0 and int(u.get("calls", 0)) >= n
    except Exception:
        return False

def load_done_tids(jsonl_path: Path) -> set[str]:
    done = set()
    if jsonl_path.exists():
        for ln in open(jsonl_path):
            ln = ln.strip()
            if not ln:
                continue
            try:
                done.add(json.loads(ln)["task_id"])
            except Exception:
                pass
    return done

try:
    t_all = time.time()
    work = Path("/content/e1_work")
    work.mkdir(parents=True, exist_ok=True)
    parts = sorted(Path("/content").glob("e1_bundle.part*"))
    if parts:
        with open("/content/e1_bundle.tgz", "wb") as fout:
            for pp in parts:
                fout.write(pp.read_bytes())
                pp.unlink(missing_ok=True)
    if parts or not (work / "data/v2/tasks.jsonl").exists():
        with tarfile.open("/content/e1_bundle.tgz", "r:gz") as tf:
            tf.extractall(work)
    os.chdir(work)
    sys.path.insert(0, str(work))

    out_dir = Path("/content/e1_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = "__MODEL_ID__"
    rel_tol = float("__REL_TOL__")
    stages = set("__STAGES__".split(","))
    K = 16
    n_reps = 6

    from benchmarks.gsb.harness import matches, score as gsb_score
    from meta_rsi.evaluator import _run_metamorphic_check
    from meta_rsi.harnesses.seed import SYSTEM_PROMPT as SEED_SYSTEM_PROMPT
    from meta_rsi.sandbox import parse_final_number, run_python_sandbox
    from meta_rsi.training.collect_trajectories import load_probes_from_dir, select_train_slice
    from meta_rsi.training.split_two_timescale import _generate_vllm_group, extract_code_block
    from scripts.rebuttal.run_e1_bok import build_matched_arms

    train_tasks = select_train_slice(Path("data/v2"), train_offset=20, train_per_family=40)
    task_meta = {t["task_id"]: t for t in train_tasks}
    probes_e1 = load_probes_from_dir(Path("scripts/rebuttal/probes_e1"), out_dir / "_probe_mods_e1")
    raw_path = out_dir / "e1_raw_candidates.jsonl"
    scored_path = out_dir / "e1_scored_candidates.jsonl"
    arms_dir = out_dir / "arms"

    # Stage 1: Sample K=16 candidates per task (resumable per task)
    if "sample" in stages:
        done_tids = load_done_tids(raw_path)
        todo = [t for t in train_tasks if t["task_id"] not in done_tids]
        if todo:
            start_vllm(model_id, out_dir / "vllm_e1_sample.log")
            print(f"[e1-bok] Sampling K={K} for {len(todo)}/{len(train_tasks)} tasks (temp=0.8, max_tokens=2048)...", flush=True)
            def _sample(t):
                comps = _generate_vllm_group(
                    "http://127.0.0.1:8000/v1", model_id, SEED_SYSTEM_PROMPT, t["prompt"],
                    n=K, temperature=0.8, max_tokens=2048, timeout=240.0,
                )
                return {"task_id": t["task_id"], "family": t["family"], "completions": comps}
            with open(raw_path, "a") as fh:
                with ThreadPoolExecutor(max_workers=16) as ex:
                    for idx, rec in enumerate(ex.map(_sample, todo), 1):
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        if idx % 50 == 0 or idx == len(todo):
                            print(f"[e1-bok] sampled {idx}/{len(todo)} tasks ({time.time()-t_all:.0f}s)", flush=True)
            kill_vllm()
        else:
            print(f"[e1-bok] Stage 1 SKIP: all {len(train_tasks)} tasks already sampled", flush=True)

    # Stage 2: Score all candidates incrementally on CPU (append per task, resumable!)
    if "score" in stages:
        done_sc = load_done_tids(scored_path)
        raw_by_tid = {}
        for ln in open(raw_path):
            if ln.strip():
                r = json.loads(ln)
                raw_by_tid[r["task_id"]] = r
        todo_rows = [raw_by_tid[t["task_id"]] for t in train_tasks if t["task_id"] in raw_by_tid and t["task_id"] not in done_sc]
        if todo_rows:
            print(f"[e1-bok] Scoring {len(todo_rows)} remaining tasks x K={K} on CPU (E1 gate rel_tol={rel_tol})...", flush=True)
            def _score_task(row):
                t = task_meta[row["task_id"]]
                low_info = t["variants"]["low"]
                low_dir = Path("data/v2") / low_info["dir"]
                out_cands = []
                for comp in row["completions"]:
                    code = extract_code_block(comp)
                    if not code:
                        continue
                    r = run_python_sandbox(code, low_dir, timeout=20.0)
                    if r["timed_out"] or r["returncode"] != 0:
                        continue
                    val = parse_final_number(r["stdout"])
                    if val is None:
                        continue
                    low_ok = bool(matches(val, low_info["truth"], t["answer"]))
                    if not low_ok:
                        continue
                    m_ok, _, _ = _run_metamorphic_check(
                        code, low_dir, val, t["answer"],
                        timeout=20.0, include_domain_hint=False, custom_probes=probes_e1, rel_tol=rel_tol,
                    )
                    ora = gsb_score(t, code, Path("data/v2"), timeout=20.0)
                    out_cands.append({
                        "code": code,
                        "low_ok": True,
                        "meta_ok": bool(m_ok),
                        "oracle_ok": bool(ora["category"] == "correct"),
                    })
                return {"task_id": row["task_id"], "family": row["family"], "candidates": out_cands}

            with open(scored_path, "a") as fh:
                with ThreadPoolExecutor(max_workers=32) as ex:
                    for idx, sr in enumerate(ex.map(_score_task, todo_rows), 1):
                        fh.write(json.dumps(sr) + "\n")
                        fh.flush()
                        if idx % 25 == 0 or idx == len(todo_rows):
                            print(f"[e1-bok] scored {idx}/{len(todo_rows)} tasks ({time.time()-t_all:.0f}s)", flush=True)
        else:
            print(f"[e1-bok] Stage 2 SKIP: all {len(train_tasks)} tasks already scored", flush=True)

        cands_by_tid = {}
        for ln in open(scored_path):
            if ln.strip():
                r = json.loads(ln)
                cands_by_tid[r["task_id"]] = r["candidates"]
        audit = build_matched_arms(cands_by_tid, task_meta, system_prompt=SEED_SYSTEM_PROMPT, replicates=n_reps, out_dir=arms_dir)
        audit["gate_rel_tol"] = rel_tol
        (arms_dir / "e1_selection_audit.json").write_text(json.dumps(audit, indent=2))
        print("[e1-bok] Selection audit:", json.dumps(audit, indent=2), flush=True)

    # Stage 3: Train 12 LoRAs (6 replicates x 2 arms, separate training seeds 42..47)
    if "train" in stages:
        for rep in range(n_reps):
            seed = 42 + rep
            w_name, m_name = f"bok_weak_r{rep}", f"bok_meta_r{rep}"
            if (out_dir / w_name / "adapter_config.json").exists() and (out_dir / m_name / "adapter_config.json").exists():
                print(f"[e1-bok] Stage 3 SKIP replicate {rep} (already trained)", flush=True)
                continue
            print(f"[e1-bok] Training replicate {rep}/{n_reps} ({w_name}, {m_name}, seed={seed})...", flush=True)
            subprocess.run(
                [
                    sys.executable, "-u", "-m", "meta_rsi.training.split_two_timescale",
                    "--mode", "sft",
                    "--model", model_id,
                    "--out-dir", str(out_dir),
                    "--data-root", "data/v2",
                    "--sft-jsonl", str(arms_dir / f"{w_name}.jsonl"),
                    "--sft-lora-name", w_name,
                    "--extra-sft", f"{m_name}={arms_dir / f'{m_name}.jsonl'}",
                    "--sft-epochs", "2",
                    "--sft-lr", "1e-4",
                    "--sft-batch-size", "16",
                    "--lora-r", "16",
                    "--lora-alpha", "32",
                    "--max-seq-len", "2560",
                    "--sync-url", "",
                    "--seed", str(seed),
                ],
                check=True,
            )

    # Stage 4: Evaluate bare base model + all 12 LoRAs on SAME server in two clean 6-LoRA batches
    if "eval" in stages:
        eval_dir = out_dir / "evals"
        eval_dir.mkdir(exist_ok=True)
        for batch_idx, rep_chunk in enumerate((range(0, 3), range(3, 6))):
            lora_names = [f"bok_{arm}_r{rep}" for rep in rep_chunk for arm in ("weak", "meta")]
            models_to_eval = ([("base_26b", model_id)] if batch_idx == 0 else []) + [(n, n) for n in lora_names]
            # Check if anything in this batch still needs running
            pending = []
            for m_tag, m_name in models_to_eval:
                for tag, data_root, proto, extra, min_n in [
                    ("protA", "data/v2", "A", [], 450),
                    ("protB100", "data/v2", "B", ["--seen-per-family", "5", "--heldout-per-family", "5"], 100),
                    ("osm_global", "data/osm_global", "A", [], 96),
                ]:
                    dst = eval_dir / f"eval_{tag}_{m_tag}.json"
                    if not is_clean(dst, min_n):
                        pending.append((m_tag, m_name, tag, data_root, proto, extra, dst, min_n))
            if not pending:
                print(f"[e1-bok] Stage 4 batch {batch_idx} SKIP (all evals clean)", flush=True)
                continue
            start_vllm(model_id, out_dir / f"vllm_e1_eval_b{batch_idx}.log", lora_modules=[(n, out_dir / n) for n in lora_names])
            for m_tag, m_name, tag, data_root, proto, extra, dst, min_n in pending:
                print(f"[e1-bok] Eval {dst.name} (model={m_name})...", flush=True)
                subprocess.run(
                    [
                        sys.executable, "-u", "-m", "meta_rsi.eval.eval_test",
                        "--harness", "meta_rsi/harnesses/seed.py",
                        "--data-root", data_root,
                        "--protocol", proto,
                        "--out", str(dst),
                        "--task-backend", "openai-compat",
                        "--task-model", m_name,
                        "--task-url", "http://127.0.0.1:8000/v1",
                        "--task-temp", "0.0",
                        "--task-max-tokens", "2048",
                        "--task-no-think",
                        "--max-llm-calls", "1",
                        "--exec-timeout", "25.0",
                        "--workers", "32",
                        *extra,
                    ],
                    check=True,
                )
            kill_vllm()

    # Stage 5: Pack evals + audits + arms cleanly (lightweight ~8 MB package!)
    if "pack" in stages:
        tgz = Path("/content/e1_evals_out.tgz")
        with tarfile.open(tgz, "w:gz") as tf:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file() and p.suffix in (".json", ".jsonl") and "_probe_mods" not in p.parts:
                    if p.name == "e1_raw_candidates.jsonl":
                        continue
                    tf.add(p, arcname=str(p.relative_to(out_dir)))
        print(f"[e1-bok] PACKED {tgz} ({tgz.stat().st_size/1e6:.2f} MB) in {time.time()-t_all:.0f}s!", flush=True)
finally:
    stop_hb.set()
'''


def dry_run_cpu() -> None:
    """Verify coupled single-RNG selection & arm-matching invariants on CPU (zero GPU)."""
    from meta_rsi.harnesses.seed import SYSTEM_PROMPT as SEED_SYSTEM_PROMPT
    from meta_rsi.training.collect_trajectories import select_train_slice

    tasks = select_train_slice(REPO / "data/v2", train_offset=20, train_per_family=40)
    assert len(tasks) == 600, len(tasks)
    task_meta = {t["task_id"]: t for t in tasks}
    cands = {}
    for i, t in enumerate(tasks):
        cs = []
        if i % 3 == 0:
            # All low_ok candidates pass meta_ok -> coupled RNG MUST pick identical scripts!
            cs.append({"code": f"print({i} + 10) # clean A", "low_ok": True, "meta_ok": True, "oracle_ok": True})
            cs.append({"code": f"print({i} + 11) # clean B", "low_ok": True, "meta_ok": True, "oracle_ok": True})
        elif i % 2 == 0:
            cs.append({"code": f"print({i} + 1) # weak shortcut", "low_ok": True, "meta_ok": False, "oracle_ok": False})
            cs.append({"code": f"print({i} + 2) # meta clean", "low_ok": True, "meta_ok": True, "oracle_ok": True})
        else:
            cs.append({"code": f"print({i} + 1) # weak only", "low_ok": True, "meta_ok": False, "oracle_ok": False})
        cands[t["task_id"]] = cs

    with tempfile.TemporaryDirectory() as td:
        audit = build_matched_arms(cands, task_meta, system_prompt=SEED_SYSTEM_PROMPT, out_dir=Path(td))
        assert len(audit["replicates"]) == 6
        assert audit["n_tasks_T"] == 400
        for rep in range(6):
            w = [json.loads(ln) for ln in open(Path(td) / f"bok_weak_r{rep}.jsonl")]
            m = [json.loads(ln) for ln in open(Path(td) / f"bok_meta_r{rep}.jsonl")]
            assert [r["task_id"] for r in w] == [r["task_id"] for r in m]
            assert all(r["meta_ok"] for r in m)
            assert all(r["low_ok"] for r in w)
            # Coupled invariant: if weak pick passed meta_ok, weak and meta MUST pick the exact same script!
            for rw, rm in zip(w, m):
                if rw["meta_ok"]:
                    assert rw["assistant"] == rm["assistant"], "Coupled RNG violated!"
    print("[e1-bok] CPU dry-run (6 reps, coupled RNG invariant verified) PASSED:", json.dumps(audit["replicates"], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 Teacher-Free Best-of-K Self-Distillation")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--session", default="")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--rel-tol", type=float, default=0.02, help="Gate agreement tolerance (0.02 per Amendment 3)")
    ap.add_argument("--stages", default="sample,score,train,eval,pack", help="Comma-separated subset of stages to run")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()

    if a.dry_run_cpu:
        dry_run_cpu()
        return
    if not a.session:
        raise SystemExit("Pass --session <colab-session> (ONLY after unified_6seed_eval & Amendment 2!) or --dry-run-cpu")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.download_only:
        local_tgz = out_dir / "e1_evals_out.tgz"
        _run_colab(["download", "-s", a.session, "/content/e1_evals_out.tgz", str(local_tgz)])
        with tarfile.open(local_tgz, "r:gz") as tf:
            tf.extractall(out_dir)
        print(f"[e1-bok] Downloaded & extracted E1 results to {out_dir}")
        return

    v2_root = REPO / "data/v2"
    rows = [json.loads(ln) for ln in open(v2_root / "tasks.jsonl")]
    by_fam_tr: dict[str, list[dict]] = {}
    test_rows = []
    for r in rows:
        if r["split"] == "train":
            by_fam_tr.setdefault(r["family"], []).append(r)
        elif r["split"] == "test":
            test_rows.append(r)
    sel = [r for fam in sorted(by_fam_tr) for r in by_fam_tr[fam][:60]] + test_rows

    with tempfile.TemporaryDirectory() as tmp:
        if not a.skip_upload:
            bundle = Path(tmp) / "e1_bundle.tgz"
            with tarfile.open(bundle, "w:gz") as tf:
                for top in ("meta_rsi", "benchmarks", "scripts/rebuttal"):
                    for p in (REPO / top).rglob("*.py"):
                        if "__pycache__" not in p.parts:
                            tf.add(p, arcname=str(p.relative_to(REPO)))
                tjsonl = Path(tmp) / "tasks.jsonl"
                tjsonl.write_text("\n".join(json.dumps(r) for r in sel) + "\n")
                tf.add(tjsonl, arcname="data/v2/tasks.jsonl")
                dirs = {v["dir"] for r in sel for v in r["variants"].values()}
                for d in sorted(dirs):
                    tf.add(v2_root / d, arcname=f"data/v2/{d}")
                if (REPO / "data/osm_global").exists():
                    tf.add(REPO / "data/osm_global", arcname="data/osm_global")
            print(f"[e1-bok] Packed bundle: {bundle.stat().st_size/1e6:.1f} MB", flush=True)
            upload_chunked(a.session, bundle, "/content/e1_bundle.part")
        cell = Path(tmp) / "remote_e1.py"
        cell.write_text(
            REMOTE_E1_DRIVER
            .replace("__MODEL_ID__", MODEL_ID)
            .replace("__REL_TOL__", str(a.rel_tol))
            .replace("__STAGES__", a.stages)
        )
        _run_colab(["exec", "-s", a.session, "--timeout", "10800", "-f", str(cell)])
        if "pack" in a.stages.split(","):
            local_tgz = out_dir / "e1_evals_out.tgz"
            _run_colab(["download", "-s", a.session, "/content/e1_evals_out.tgz", str(local_tgz)])
            with tarfile.open(local_tgz, "r:gz") as tf:
                tf.extractall(out_dir)
            print(f"[e1-bok] Extracted all E1 results to {out_dir}")


if __name__ == "__main__":
    main()

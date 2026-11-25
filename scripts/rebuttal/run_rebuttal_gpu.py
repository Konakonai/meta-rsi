"""Single-GPU Rebuttal Master Runner (Plan B: M3 + M4 + M5 + M6).

Executes ALL Plan B GPU experiments cleanly on a single Colab G4 (96 GB) or H100 (80 GB) session:
  Stage 0: Environment setup (geospatial + vLLM + PEFT, remove conflicting torchao) & upload existing
           Wave S4-2 LoRAs (`sft_weak`, `sft_meta`, `grpo_meta_final`), `data/v2`, `data/osm_global`,
           and certified probes via chunked upload.
  Stage 1 (M4 — De-confounded Trajectory Collection, disjoint train[20:60] = 600 tasks):
           - `same_teacher_weak`: `stage2_coev_best.py` teacher + `weak` filter (eliminates teacher confound vs `sft_meta`)
           - `base_weak` & `base_meta`: bare `seed.py` (ZERO evolved teacher) + `weak` vs `metamorphic` 3-probe filter
  Stage 2 (M4 — De-confounded SFT + RL Controls, exact BF16 parity with Wave S4-2):
           - `sft_same_teacher_weak` (r=16, 2 ep on `same_teacher_weak`)
           - `sft_base_weak` & `sft_base_meta` (r=16, 2 ep on bare-model trajectories)
           - `sft_meta_more` (r=16, 5 ep on `meta_trajectories.jsonl`): extended-SFT control for GRPO
             (more SFT epochs on the same data; NOT compute-matched to GRPO's 768 rollouts)
           - `grpo_weak_final` (initialized from EXACT SAME `sft_meta`, 4 rounds x 6 steps = 24 steps x 32 rollouts = 768,
                                `reward_mode="weak"` — directly tests whether weak RL re-induces weight-level proxy lock-in!)
  Stage 3 (M3 + M4 + M5 + M6 — Multi-Protocol Evaluations, bare `seed.py`, 1.00 call/task):
           - Protocol A (`eval450`): all 5 new adapters (`sft_same_teacher_weak`, `sft_base_weak`,
             `sft_base_meta`, `sft_meta_more`, `grpo_weak_final`)
           - 3x Repeat BF16 Variance Audit (`eval450` Protocol A, 3 independent runs each):
             Base 26B, `sft_meta`, `grpo_meta_final`
           - Protocol B (`eval450` live per-variant deployment, real high/polar/antimeridian visible!):
             Base 26B, `sft_weak`, `sft_same_teacher_weak`, `sft_meta`, `grpo_weak_final`, `grpo_meta_final`
           - Global Real OSM Executable Suite (`data/osm_global`, 8 cities x 12 tasks = 96 tasks incl.
             Arctic 78N, Southern Hemisphere, & 16 Absolute-Position non-invariant tasks; Wellington at
             174.8E does not cross 180°, so the antimeridian is not exercised on real data):
             Base 26B, `sft_weak`, `sft_same_teacher_weak`, `sft_meta`, `grpo_weak_final`, `grpo_meta_final`
  Stage 4 (M3 — Balanced 6-Seed GSB v2 Table):
           - Fill any remaining seeds `s1..s5` of `naive`, `meta_gate_only`, `oracle_soft` on fast local vLLM!
  Stage 5: Pack & download all LoRAs, summaries, and strictly audited (`uerr==0 && calls>=n`) JSONs to
           `runs/v9_rebuttal_gpu/`.

Usage:
    nix-shell --run "python -u scripts/rebuttal/run_rebuttal_gpu.py --session w3-rebuttal"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.wave_s4_2.run_single_g4_wave2 import (
    _run_colab,
    upload_chunked,
)


def download_rebuttal_chunked(session: str, out_tgz: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest_p = Path(tmp) / "manifest.json"
        _run_colab(["download", "-s", session, "/content/rebuttal_out_manifest.json", str(manifest_p)])
        manifest = json.loads(manifest_p.read_text())
        with open(out_tgz, "wb") as fout:
            for pname in manifest["parts"]:
                lp = Path(tmp) / pname
                _run_colab(["download", "-s", session, f"/content/{pname}", str(lp)])
                fout.write(lp.read_bytes())

MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_OUT_DIR = REPO / "runs/v9_rebuttal_gpu"

REMOTE_REBUTTAL_DRIVER = r'''
import json, os, random, shutil, subprocess, sys, tarfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

stop_hb = threading.Event()
def _hb():
    t0 = time.time()
    while not stop_hb.wait(20.0):
        print(f"[reb-gpu hb] elapsed={time.time()-t0:.0f}s", flush=True)
threading.Thread(target=_hb, daemon=True).start()

def kill_vllm():
    subprocess.run(["pkill", "-9", "-f", "vllm|VLLM|EngineCore"], check=False)
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
            casted = {
                k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype != torch.bfloat16 else v)
                for k, v in tensors.items()
            }
            save_file(casted, str(dst))
        else:
            shutil.copy2(src, dst)
    return dst_dir

def start_vllm(model: str, log_path: Path, lora_modules: list[tuple[str, Path]] | None = None, gpu_mem: str = "0.86") -> subprocess.Popen:
    kill_vllm()
    env = os.environ.copy()
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    log_f = open(log_path, "w")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--gpu-memory-utilization", gpu_mem,
        "--max-model-len", "8192",
        "--max-num-seqs", "64",
        "--enable-lora",
        "--max-loras", "8",
        "--max-lora-rank", "16",
    ]
    if lora_modules:
        cmd.append("--lora-modules")
        for lname, ldir in lora_modules:
            bf_dir = cast_lora_bf16(ldir, Path("/content/loras_bf16") / lname)
            cmd.append(f"{lname}={bf_dir}")
    print(f"[reb-gpu] launching vLLM ({log_path.name}, loras={[x[0] for x in (lora_modules or [])]})...", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while not http_ok("http://127.0.0.1:8000/v1/models"):
        if proc.poll() is not None:
            print(log_path.read_text()[-3000:], file=sys.stderr, flush=True)
            raise RuntimeError(f"vLLM exited with rc={proc.returncode}")
        if time.time() - t0 > 600:
            raise TimeoutError("Timed out waiting for vLLM")
        time.sleep(5)
    print(f"[reb-gpu] vLLM READY in {time.time()-t0:.1f}s", flush=True)
    return proc

def is_clean_eval(p: Path, min_n: int) -> bool:
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        u = d.get("usage") or {}
        n = int((d.get("ALL") or {}).get("n", 0))
        return n >= min_n and int(u.get("errors", 1)) == 0 and int(u.get("calls", 0)) >= n
    except Exception:
        return False

def run_eval_test(model_name: str, data_root: str, out_json: Path, protocol: str = "A", workers: int = 24, harness: str = "meta_rsi/harnesses/seed.py", max_calls: int = 1, min_n: int = 1) -> None:
    if is_clean_eval(out_json, min_n):
        d = json.loads(out_json.read_text())
        print(f"[reb-gpu] SKIP clean {out_json.name}: ALL={d['ALL']['correct']*100:.1f}%", flush=True)
        return
    subprocess.run(
        [
            sys.executable, "-u", "-m", "meta_rsi.eval.eval_test",
            "--harness", harness,
            "--data-root", data_root,
            "--protocol", protocol,
            "--out", str(out_json),
            "--task-backend", "openai-compat",
            "--task-model", model_name,
            "--task-url", "http://127.0.0.1:8000/v1",
            "--task-temp", "0.0",
            "--task-max-tokens", "2048",
            "--task-no-think",
            "--max-llm-calls", str(max_calls),
            "--exec-timeout", "25.0",
            "--workers", str(workers),
        ],
        check=True,
    )
    if not is_clean_eval(out_json, min_n):
        raise RuntimeError(f"Audit failed on {out_json}")

try:
    t_start = time.time()
    work = Path("/content/rebuttal_work")
    work.mkdir(parents=True, exist_ok=True)
    parts = sorted(Path("/content").glob("rebuttal_bundle.part*"))
    if parts:
        with open("/content/rebuttal_bundle.tgz", "wb") as fout:
            for pp in parts:
                fout.write(pp.read_bytes())
                pp.unlink(missing_ok=True)
        print(f"[reb-gpu] reassembled {len(parts)} bundle chunks", flush=True)
    if parts or not (work / "data/v2/tasks.jsonl").exists():
        with tarfile.open("/content/rebuttal_bundle.tgz", "r:gz") as tf:
            tf.extractall(work)
    os.chdir(work)
    sys.path.insert(0, str(work))

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)

    # Stage 0: Dependencies
    need = []
    for mod, pkg in [
        ("geopandas", "geopandas"), ("pyogrio", "pyogrio"), ("pyproj", "pyproj"),
        ("shapely", "shapely"), ("rasterio", "rasterio"), ("scipy", "scipy"),
        ("peft", "peft"), ("vllm", "vllm"),
    ]:
        try:
            __import__(mod)
        except Exception:
            need.append(pkg)
    if need:
        print(f"[reb-gpu] installing missing packages: {need}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

    out_dir = Path("/content/rebuttal_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = work / "rebuttal_assets/certified_probes"
    prev_loras = work / "rebuttal_assets/prev_loras"
    prev_meta_traj = work / "rebuttal_assets/meta_trajectories.jsonl"
    meta_teacher = work / "rebuttal_assets/meta_teacher.py"
    model_id = "__MODEL_ID__"

    # ------------------------------------------------------------------
    # STAGE 1: De-confounded Trajectory Collection (train[20:60] -> offset=0 on pre-sliced bundle!)
    # ------------------------------------------------------------------
    st_weak_jsonl = traj_dir / "same_teacher_weak.jsonl"
    base_weak_jsonl = traj_dir / "base_weak.jsonl"
    base_meta_jsonl = traj_dir / "base_meta.jsonl"

    if not (st_weak_jsonl.exists() and base_weak_jsonl.exists() and base_meta_jsonl.exists()):
        if not http_ok("http://127.0.0.1:8000/v1/models"):
            start_vllm(model_id, out_dir / "vllm_stage1.log")
        else:
            print("[reb-gpu] Reusing already-running vLLM server on :8000!", flush=True)
        if not st_weak_jsonl.exists():
            print("[reb-gpu] Collecting same_teacher_weak trajectories (stage2_coev + weak filter)...", flush=True)
            subprocess.run(
                [
                    sys.executable, "-u", "-m", "meta_rsi.training.collect_trajectories",
                    "--harness", str(meta_teacher),
                    "--data-root", "data/v2",
                    "--out", str(st_weak_jsonl),
                    "--filter-mode", "weak",
                    "--train-offset", "0",
                    "--train-per-family", "40",
                    "--task-backend", "openai-compat",
                    "--task-model", model_id,
                    "--task-url", "http://127.0.0.1:8000/v1",
                    "--task-temp", "0.0",
                    "--task-max-tokens", "2048",
                    "--task-no-think",
                    "--exec-timeout", "25.0",
                    "--workers", "16",
                ],
                check=True,
            )
        if not base_weak_jsonl.exists():
            print("[reb-gpu] Collecting base_weak trajectories (bare seed.py + weak filter)...", flush=True)
            subprocess.run(
                [
                    sys.executable, "-u", "-m", "meta_rsi.training.collect_trajectories",
                    "--harness", "meta_rsi/harnesses/seed.py",
                    "--data-root", "data/v2",
                    "--out", str(base_weak_jsonl),
                    "--filter-mode", "weak",
                    "--train-offset", "0",
                    "--train-per-family", "40",
                    "--task-backend", "openai-compat",
                    "--task-model", model_id,
                    "--task-url", "http://127.0.0.1:8000/v1",
                    "--task-temp", "0.0",
                    "--task-max-tokens", "2048",
                    "--task-no-think",
                    "--exec-timeout", "25.0",
                    "--workers", "16",
                ],
                check=True,
            )
        if not base_meta_jsonl.exists():
            print("[reb-gpu] Collecting base_meta trajectories (bare seed.py + metamorphic 3-probe filter)...", flush=True)
            subprocess.run(
                [
                    sys.executable, "-u", "-m", "meta_rsi.training.collect_trajectories",
                    "--harness", "meta_rsi/harnesses/seed.py",
                    "--data-root", "data/v2",
                    "--out", str(base_meta_jsonl),
                    "--filter-mode", "metamorphic",
                    "--probe-dir", str(probe_dir),
                    "--train-offset", "0",
                    "--train-per-family", "40",
                    "--task-backend", "openai-compat",
                    "--task-model", model_id,
                    "--task-url", "http://127.0.0.1:8000/v1",
                    "--task-temp", "0.0",
                    "--task-max-tokens", "2048",
                    "--task-no-think",
                    "--exec-timeout", "25.0",
                    "--workers", "16",
                ],
                check=True,
            )
        kill_vllm()

    # ------------------------------------------------------------------
    # STAGE 2A: Train 3 de-confounded 2-epoch SFT adapters in ONE model load + 5-epoch SFT-More
    # ------------------------------------------------------------------
    sft_st_weak_dir = out_dir / "sft_same_teacher_weak"
    sft_base_weak_dir = out_dir / "sft_base_weak"
    sft_base_meta_dir = out_dir / "sft_base_meta"
    sft_meta_more_dir = out_dir / "sft_meta_more"

    if not ((sft_st_weak_dir / "adapter_config.json").exists() and (sft_base_weak_dir / "adapter_config.json").exists() and (sft_base_meta_dir / "adapter_config.json").exists()):
        print("[reb-gpu] Training 3 de-confounded 2-epoch SFT LoRAs in single bf16 load...", flush=True)
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.training.split_two_timescale",
                "--mode", "sft",
                "--model", model_id,
                "--out-dir", str(out_dir),
                "--data-root", "data/v2",
                "--sft-jsonl", str(st_weak_jsonl),
                "--sft-lora-name", "sft_same_teacher_weak",
                "--extra-sft", f"sft_base_weak={base_weak_jsonl}",
                "--extra-sft", f"sft_base_meta={base_meta_jsonl}",
                "--sft-epochs", "2",
                "--sft-lr", "1e-4",
                "--sft-batch-size", "16",
                "--lora-r", "16",
                "--lora-alpha", "32",
                "--max-seq-len", "2560",
                "--sync-url", "",
                "--seed", "42",
            ],
            check=True,
        )

    if not (sft_meta_more_dir / "adapter_config.json").exists():
        print("[reb-gpu] Training extended SFT-More control (5 epochs on meta_trajectories.jsonl)...", flush=True)
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.training.split_two_timescale",
                "--mode", "sft",
                "--model", model_id,
                "--out-dir", str(out_dir),
                "--data-root", "data/v2",
                "--sft-jsonl", str(prev_meta_traj),
                "--sft-lora-name", "sft_meta_more",
                "--sft-epochs", "5",
                "--sft-lr", "1e-4",
                "--sft-batch-size", "16",
                "--lora-r", "16",
                "--lora-alpha", "32",
                "--max-seq-len", "2560",
                "--sync-url", "",
                "--seed", "42",
            ],
            check=True,
        )

    # ------------------------------------------------------------------
    # STAGE 2B: Same-init GRPO-Weak control (EXACT parity with GRPO-Meta, ONLY reward_mode="weak"!)
    # ------------------------------------------------------------------
    from meta_rsi.harnesses.seed import SYSTEM_PROMPT as SEED_SYSTEM_PROMPT
    from meta_rsi.training.collect_trajectories import select_train_slice
    from meta_rsi.training.split_two_timescale import _generate_vllm_group, score_completion

    sft_meta_dir = prev_loras / "sft_meta"
    grpo_weak_final = out_dir / "grpo_weak_final"
    if not (grpo_weak_final / "adapter_config.json").exists():
        all_tasks = select_train_slice(Path("data/v2"), train_offset=0, train_per_family=40)
        rng = random.Random(42)
        n_rounds = 4
        steps_per_round = 6
        tasks_per_step = 4
        group_size = 8
        cur_policy_dir = sft_meta_dir
        opt_state_path = out_dir / "grpo_weak_opt.pt"
        all_step_logs = []
        step_cursor = 0

        for rd in range(1, n_rounds + 1):
            round_batches = [
                rng.sample(all_tasks, min(tasks_per_step, len(all_tasks)))
                for _ in range(steps_per_round)
            ]
            save_name = "grpo_weak_final" if rd == n_rounds else f"grpo_weak_r{rd}"
            if (out_dir / save_name / "adapter_config.json").exists() and (out_dir / f"{save_name}_summary.json").exists():
                print(f"[grpo-weak] Round {rd} already done, skipping", flush=True)
                rs = json.loads((out_dir / f"{save_name}_summary.json").read_text())
                all_step_logs.extend(rs["steps"])
                cur_policy_dir = out_dir / save_name
                step_cursor += steps_per_round
                continue
            lora_name = f"grpo_weak_r{rd-1}"
            start_vllm(model_id, out_dir / f"vllm_grpo_weak_r{rd}.log", lora_modules=[(lora_name, cur_policy_dir)])
            rollouts_jsonl = out_dir / f"grpo_weak_round{rd}_rollouts.jsonl"
            rows = []
            for batch in round_batches:
                step_cursor += 1
                t_gen = time.time()
                def _gen(t):
                    return t, _generate_vllm_group(
                        "http://127.0.0.1:8000/v1", lora_name, SEED_SYSTEM_PROMPT, t["prompt"],
                        n=group_size, temperature=0.8, max_tokens=1536,
                    )
                with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                    gen_out = list(ex.map(_gen, batch))
                gen_s = time.time() - t_gen
                t_sc = time.time()
                score_jobs = [(t, comp) for t, comps in gen_out for comp in comps]
                def _sc(pair):
                    t, comp = pair
                    r, cat, u3857 = score_completion(
                        comp, t, Path("data/v2"), reward_mode="weak", custom_probes=None, timeout=25.0
                    )
                    return {"step": step_cursor, "task_id": t["task_id"], "prompt": t["prompt"],
                            "completion": comp, "reward": r, "cat": cat, "uses_3857": u3857}
                with ThreadPoolExecutor(max_workers=24) as ex:
                    step_rows = list(ex.map(_sc, score_jobs))
                sc_s = time.time() - t_sc
                n_full = sum(1 for x in step_rows if x["reward"] >= 0.99)
                n3857 = sum(1 for x in step_rows if x["uses_3857"])
                print(
                    f"[grpo-weak r{rd}] step={step_cursor:2d} mean_r={sum(x['reward'] for x in step_rows)/len(step_rows):.3f} "
                    f"full={n_full:2d}/{len(step_rows)} 3857={n3857:2d} (gen={gen_s:.1f}s sc={sc_s:.1f}s)",
                    flush=True,
                )
                rows.extend(step_rows)
            with open(rollouts_jsonl, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            kill_vllm()

            subprocess.run(
                [
                    sys.executable, "-u", "-m", "meta_rsi.training.split_two_timescale",
                    "--mode", "grpo_offline_batch",
                    "--model", model_id,
                    "--out-dir", str(out_dir),
                    "--data-root", "data/v2",
                    "--sft-lora-dir", str(sft_meta_dir),
                    "--init-lora-dir", str(cur_policy_dir),
                    "--rollouts-jsonl", str(rollouts_jsonl),
                    "--save-lora-name", save_name,
                    "--opt-state-path", str(opt_state_path),
                    "--grpo-lr", "1e-5",
                    "--grpo-beta", "0.02",
                    "--max-seq-len", "2560",
                    "--sync-url", "",
                    "--seed", str(42 + rd),
                ],
                check=True,
            )
            cur_policy_dir = out_dir / save_name
            rs = json.loads((out_dir / f"{save_name}_summary.json").read_text())
            all_step_logs.extend(rs["steps"])

        (out_dir / "grpo_weak_combined_summary.json").write_text(
            json.dumps({"stage": "grpo_weak_alternating", "n_rounds": n_rounds, "steps": all_step_logs}, indent=2)
        )

    # ------------------------------------------------------------------
    # STAGE 3: All Multi-Protocol Evaluations in ONE vLLM Server Load!
    # ------------------------------------------------------------------
    eval_loras = [
        ("sft_weak", prev_loras / "sft_weak"),
        ("sft_meta", prev_loras / "sft_meta"),
        ("grpo_meta_final", prev_loras / "grpo_meta_final"),
        ("sft_same_teacher_weak", sft_st_weak_dir),
        ("sft_base_weak", sft_base_weak_dir),
        ("sft_base_meta", sft_base_meta_dir),
        ("sft_meta_more", sft_meta_more_dir),
        ("grpo_weak_final", grpo_weak_final),
    ]
    start_vllm(model_id, out_dir / "vllm_stage3_eval.log", lora_modules=eval_loras)

    # 3A: Protocol A (eval450) for all 5 new adapters
    for lname in ("sft_same_teacher_weak", "sft_base_weak", "sft_base_meta", "sft_meta_more", "grpo_weak_final"):
        run_eval_test(lname, "data/v2", out_dir / f"eval_v2_protA_{lname}.json", protocol="A", workers=24, min_n=450)

    # 3B: 3x Repeat BF16 Variance Audit (eval450 Protocol A)
    for m_tag, m_name in [("base_26b", model_id), ("sft_meta", "sft_meta"), ("grpo_meta_final", "grpo_meta_final")]:
        for rep in (1, 2, 3):
            run_eval_test(m_name, "data/v2", out_dir / f"eval_v2_var_{m_tag}_rep{rep}.json", protocol="A", workers=24, min_n=450)

    # 3C: Protocol B (eval450 live per-variant deployment!)
    all_9_models = [
        ("base_26b", model_id),
        ("sft_weak", "sft_weak"),
        ("sft_same_teacher_weak", "sft_same_teacher_weak"),
        ("sft_base_weak", "sft_base_weak"),
        ("sft_base_meta", "sft_base_meta"),
        ("sft_meta", "sft_meta"),
        ("sft_meta_more", "sft_meta_more"),
        ("grpo_weak_final", "grpo_weak_final"),
        ("grpo_meta_final", "grpo_meta_final"),
    ]
    for m_tag, m_name in all_9_models:
        run_eval_test(m_name, "data/v2", out_dir / f"eval_v2_protB_{m_tag}.json", protocol="B", workers=24, min_n=450)

    # 3D: Expanded Global Real OSM Executable Suite (96 tasks incl. Arctic 78N & 16 Absolute-Position tasks!)
    if Path("data/osm_global/tasks.jsonl").exists():
        for m_tag, m_name in all_9_models:
            run_eval_test(m_name, "data/osm_global", out_dir / f"eval_osm_global_{m_tag}.json", protocol="A", workers=24, min_n=96)

    # 3E: Also evaluate existing 4-city OSM on new adapters for clean 2x2 table parity!
    if Path("data/osm_eval/tasks.jsonl").exists():
        for lname in ("sft_same_teacher_weak", "sft_base_weak", "sft_base_meta", "sft_meta_more", "grpo_weak_final"):
            run_eval_test(lname, "data/osm_eval", out_dir / f"eval_osm_{lname}.json", protocol="A", workers=24, min_n=24)

    # ------------------------------------------------------------------
    # STAGE 4: Fast vLLM completion of missing 6-seed baselines (naive, meta_gate_only, oracle_soft s1..s5)
    # ------------------------------------------------------------------
    if "__RUN_MISSING_SEEDS__" == "1":
        print("[reb-gpu] Running missing 6-seed GSB v2 baselines (naive, meta_gate_only, oracle_soft s1..s5) on local vLLM...", flush=True)
        subprocess.run(
            [
                sys.executable, "-u", "scripts/rebuttal/run_v2_missing_seeds.py",
                "--backend", "vllm",
                "--task-url", "http://127.0.0.1:8000/v1",
                "--task-model", model_id,
                "--out-dir", str(out_dir / "missing_6seed"),
                "--data-root", "data/v2",
                "--probe-dir", str(probe_dir),
                "--workers", "16",
                "--eval-workers", "24",
            ],
            check=True,
        )

    kill_vllm()

    # Stage 5: Pack outputs into downloadable chunks
    out_tgz = Path("/content/rebuttal_all_out.tgz")
    with tarfile.open(out_tgz, "w:gz") as tf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and (
                p.suffix in (".json", ".jsonl", ".safetensors", ".py")
                or p.name in ("adapter_config.json", "README.md")
            ):
                rd_dir = p.parent.name  # intermediate GRPO-Weak round adapters: grpo_weak_r1 .. grpo_weak_r3
                is_round_adapter = rd_dir.startswith("grpo_weak_r") and rd_dir[len("grpo_weak_r"):].isdigit()
                if is_round_adapter or "_probe_mods" in p.parts or "evos" in p.parts:
                    continue
                tf.add(p, arcname=str(p.relative_to(out_dir)))
    raw = out_tgz.read_bytes()
    chunk_sz = 25 * 1024 * 1024
    n_parts = (len(raw) + chunk_sz - 1) // chunk_sz
    part_names = []
    for idx in range(n_parts):
        pname = f"rebuttal_all_out.part{idx:02d}"
        Path(f"/content/{pname}").write_bytes(raw[idx * chunk_sz : (idx + 1) * chunk_sz])
        part_names.append(pname)
    Path("/content/rebuttal_out_manifest.json").write_text(json.dumps({"parts": part_names, "bytes": len(raw)}, indent=2))
    print(f"[reb-gpu] ALL DONE in {time.time()-t_start:.1f}s! packed={len(raw)/1e6:.1f} MB ({n_parts} parts)", flush=True)
finally:
    stop_hb.set()
'''


def pack_rebuttal_bundle(out_tgz: Path) -> None:
    """Pack code, pre-sliced data/v2 (train[20:60] + test + train[0:12] for 6-seed evos!), data/osm_global, data/osm_eval, and existing Wave 2 LoRAs."""
    print(f"[rebuttal] Packing bundle -> {out_tgz} ...", flush=True)
    v2_root = REPO / "data/v2"
    rows = [json.loads(ln) for ln in open(v2_root / "tasks.jsonl")]
    by_fam_tr: dict[str, list[dict]] = {}
    test_rows: list[dict] = []
    for r in rows:
        if r["split"] == "train":
            by_fam_tr.setdefault(r["family"], []).append(r)
        elif r["split"] == "test":
            test_rows.append(r)

    # IMPORTANT: Keep FULL train[0:60] so BOTH dev_offset=0..10 (6-seed evos) AND train_offset=20..60 (trajectories) work cleanly!
    # Wait: in our driver, select_train_slice(..., train_offset=0) was used on pre-sliced [20:60],
    # OR if we keep train[0:60], then train_offset=20 works identically to local!
    # Let's keep train[0:60] and use train_offset=20 everywhere (ZERO offset mismatch!).
    selected_rows: list[dict] = []
    for fam in sorted(by_fam_tr):
        selected_rows.extend(by_fam_tr[fam][:60])
    selected_rows.extend(test_rows)

    with tarfile.open(out_tgz, "w:gz") as tf:
        for top in ("meta_rsi", "benchmarks", "scripts/rebuttal", "scripts/wave_s4_2"):
            base = REPO / top
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json"):
                    tf.add(p, arcname=str(p.relative_to(REPO)))

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmpf:
            for r in selected_rows:
                tmpf.write(json.dumps(r) + "\n")
            tmp_tasks = Path(tmpf.name)
        tf.add(tmp_tasks, arcname="data/v2/tasks.jsonl")
        tmp_tasks.unlink(missing_ok=True)

        added_dirs: set[str] = set()
        for r in selected_rows:
            for vinfo in r["variants"].values():
                drel = vinfo["dir"]
                if drel in added_dirs:
                    continue
                added_dirs.add(drel)
                dpath = v2_root / drel
                if dpath.exists():
                    for f in dpath.rglob("*"):
                        if f.is_file():
                            tf.add(f, arcname=f"data/v2/{f.relative_to(v2_root)}")

        for extra_data in ("data/osm_eval", "data/osm_global"):
            eroot = REPO / extra_data
            if eroot.exists():
                for f in eroot.rglob("*"):
                    if f.is_file():
                        tf.add(f, arcname=str(f.relative_to(REPO)))

        # Assets: certified probes, meta_teacher, meta_trajectories.jsonl, existing 3 Wave S4-2 LoRAs
        w2 = REPO / "runs/v8_s4_wave2_v2"
        for pf in sorted((w2 / "certified_probes").glob("*.py")):
            tf.add(pf, arcname=f"rebuttal_assets/certified_probes/{pf.name}")
        tf.add(REPO / "runs/v6_gsb_v2_s3b_stage2/stage2_coev_best.py", arcname="rebuttal_assets/meta_teacher.py")
        tf.add(w2 / "trajectories/meta_trajectories.jsonl", arcname="rebuttal_assets/meta_trajectories.jsonl")
        for lname in ("sft_weak", "sft_meta", "grpo_meta_final"):
            ldir = w2 / lname
            for f in ldir.rglob("*"):
                if f.is_file():
                    tf.add(f, arcname=f"rebuttal_assets/prev_loras/{lname}/{f.relative_to(ldir)}")

    print(f"[rebuttal] Bundle size: {out_tgz.stat().st_size / 1e6:.1f} MB (tasks={len(selected_rows)})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-GPU Plan B Rebuttal Runner")
    ap.add_argument("--session", required=True, help="Active Colab G4/H100 session name")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--skip-missing-seeds", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "rebuttal_bundle.tgz"
    if not a.skip_upload:
        pack_rebuttal_bundle(bundle)
        upload_chunked(a.session, bundle, "/content/rebuttal_bundle.part")

    # Note: bundle contains full train[0:60] so use train_offset=20!
    code = (
        REMOTE_REBUTTAL_DRIVER
        .replace("__MODEL_ID__", MODEL_ID)
        .replace("__RUN_MISSING_SEEDS__", "0" if a.skip_missing_seeds else "1")
        .replace('"--train-offset", "0"', '"--train-offset", "20"')
        .replace("train_offset=0", "train_offset=20")
    )
    print(f"[rebuttal] Launching remote Plan B GPU driver on {a.session} ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        drv = Path(tmp) / "remote_rebuttal_driver.py"
        drv.write_text(code)
        _run_colab(["exec", "-s", a.session, "--timeout", "10800", "-f", str(drv)])

    print(f"[rebuttal] Downloading outputs -> {out_dir} ...", flush=True)
    out_tgz = out_dir / "rebuttal_all_out.tgz"
    download_rebuttal_chunked(a.session, out_tgz)
    with tarfile.open(out_tgz, "r:gz") as tf:
        tf.extractall(out_dir)
    print(f"[rebuttal] Extracted all Plan B outputs into {out_dir}", flush=True)


if __name__ == "__main__":
    main()

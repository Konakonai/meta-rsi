"""Single-G4 (96 GB VRAM, 48 CPU cores, 176 GB RAM) End-to-End Wave S4-2 Runner.

Runs the COMPLETE Sprint 4 Wave S4-2 (GSB v2 Two-Timescale Weight Internalization) cleanly
on a single Colab G4 session (zero SSH tunnel dependency, 100% BF16 exact parity with v5):
  Stage 0: Install missing geospatial + vLLM + PEFT packages (uninstalling conflicting torchao).
  Stage 1: Boot vLLM on `google/gemma-4-26B-A4B-it` & collect any uncollected disjoint `train[20:60]`
           trajectories (`weak_trajectories.jsonl` + `meta_trajectories.jsonl` w/ 3 certified probes).
  Stage 2: Train `SFT-Weak` (control) + `SFT-Meta` (warm-start) BF16 LoRA adapters (`r=16`, 2 epochs).
  Stage 3: 4-round alternating online `GRPO-Metamorphic` (`4 × 6 = 24` steps × `32` rollouts = `768`
           rollouts, exact Schulman k3 KL vs frozen `sft_ref`, persistent AdamW moments, 3 probes).
  Stage 4: Load all 3 bf16 LoRAs (`sft_weak`, `sft_meta`, `grpo_meta_final`) into vLLM and evaluate
           bare `seed.py` (`1.00` call/task) on `data/v2` (`eval450`), `data/osm_eval` (`n=24`),
           and external `GeoAnalystBench` (`n=17`) — 9 strictly audited JSONs total.
  Stage 5: Pack & download all LoRAs, trajectories, training summaries, and eval JSONs to
           `runs/v8_s4_wave2_v2/`, keeping the G4 session alive by default (`--keep`).

Usage:
    nix-shell --run "python -u scripts/wave_s4_2/run_single_g4_wave2.py --session w2-train"
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

from scripts.wave_s4_2.collect_v2_trajectories import (
    DEFAULT_META_TEACHER,
    DEFAULT_OUT_DIR,
    DEFAULT_WEAK_TEACHER,
)
from scripts.wave_s4_2.run_dual_gpu_wave2 import (
    MODEL_ID,
    audit_downloaded_evals,
    pack_wave2_bundle,
)

REMOTE_SINGLE_G4_DRIVER = r'''
import json, os, random, shutil, subprocess, sys, tarfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

stop_hb = threading.Event()
def _hb():
    t0 = time.time()
    while not stop_hb.wait(15.0):
        print(f"[w2-g4 hb] elapsed={time.time()-t0:.0f}s", flush=True)
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

def start_vllm(model: str, log_path: Path, lora_modules: list[tuple[str, Path]] | None = None, gpu_mem: str = "0.88") -> subprocess.Popen:
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
        "--max-loras", "4",
        "--max-lora-rank", "16",
    ]
    if lora_modules:
        cmd.append("--lora-modules")
        for lname, ldir in lora_modules:
            bf_dir = cast_lora_bf16(ldir, Path("/content/loras_bf16") / lname)
            cmd.append(f"{lname}={bf_dir}")
    print(f"[w2-g4] launching vLLM ({log_path.name}, loras={[x[0] for x in (lora_modules or [])]})...", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while not http_ok("http://127.0.0.1:8000/v1/models"):
        if proc.poll() is not None:
            print(log_path.read_text()[-3000:], file=sys.stderr, flush=True)
            raise RuntimeError(f"vLLM exited with rc={proc.returncode}")
        if time.time() - t0 > 600:
            raise TimeoutError("Timed out waiting for vLLM")
        time.sleep(5)
    print(f"[w2-g4] vLLM READY in {time.time()-t0:.1f}s", flush=True)
    return proc

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

def load_lora_vllm(lora_name: str, lora_dir: Path) -> None:
    bf16_dir = cast_lora_bf16(lora_dir, Path("/content/loras_bf16") / lora_name)
    for ep, payload in [
        ("/v1/unload_lora_adapter", {"lora_name": lora_name}),
        ("/v1/load_lora_adapter", {"lora_name": lora_name, "lora_path": str(bf16_dir)}),
    ]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:8000{ep}", body, {"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if "load_lora" in ep:
                    print(f"[w2-g4] hot-loaded LoRA {lora_name}: HTTP {r.status}", flush=True)
        except Exception as exc:
            if "load_lora" in ep:
                raise RuntimeError(f"Failed loading LoRA {lora_name}: {exc}")

try:
    t_start = time.time()
    work = Path("/content/wave2_work")
    work.mkdir(parents=True, exist_ok=True)
    parts = sorted(Path("/content").glob("wave2_bundle.part*"))
    if parts:
        with open("/content/wave2_bundle.tgz", "wb") as fout:
            for pp in parts:
                fout.write(pp.read_bytes())
                pp.unlink(missing_ok=True)
        print(f"[w2-g4] reassembled {len(parts)} bundle chunks -> /content/wave2_bundle.tgz", flush=True)
    if not (work / "data/v2/tasks.jsonl").exists() or parts:
        with tarfile.open("/content/wave2_bundle.tgz", "r:gz") as tf:
            tf.extractall(work)
    os.chdir(work)
    sys.path.insert(0, str(work))

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)

    # Stage 0: Dependencies (uninstall conflicting torchao first!)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
    needed = []
    for mod, pkg in [
        ("vllm", "vllm"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("geopandas", "geopandas"),
        ("pyproj", "pyproj"),
        ("shapely", "shapely"),
        ("rasterio", "rasterio"),
        ("scipy", "scipy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        print("[w2-g4] installing:", needed, flush=True)
        try:
            subprocess.run(["uv", "pip", "install", "--system", *needed], check=True)
        except FileNotFoundError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)

    model = os.environ.get("W2_MODEL", "google/gemma-4-26B-A4B-it")
    # Note: bundled data/v2/tasks.jsonl is ALREADY pre-sliced to train[train_offset : train_offset+train_per_family],
    # so remote train_offset inside the bundle is always 0!
    t_off = 0
    t_pf = int(os.environ.get("W2_TRAIN_PER_FAMILY", "40"))
    sft_epochs = int(os.environ.get("W2_SFT_EPOCHS", "2"))
    grpo_rounds = int(os.environ.get("W2_GRPO_ROUNDS", "4"))
    steps_per_round = int(os.environ.get("W2_STEPS_PER_ROUND", "6"))
    tasks_per_step = 4
    num_generations = 8
    max_seq_len = int(os.environ.get("W2_MAX_SEQ_LEN", "2560"))
    max_comp_tokens = int(os.environ.get("W2_MAX_COMP_TOKENS", "1536"))

    out_dir = Path("/content/wave2_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = work / "wave2_assets" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = work / "wave2_assets" / "certified_probes"

    def _traj_has_full_count(p: Path, expected_n: int) -> bool:
        sp = p.with_suffix(".summary.json")
        if not p.exists() or not sp.exists():
            return False
        try:
            s = json.loads(sp.read_text())
            return int(s.get("n_total", 0)) >= expected_n and int((s.get("usage") or {}).get("errors", 1)) == 0
        except Exception:
            return False

    expected_n = 15 * t_pf
    # Stage 1: Collect any missing or under-counted disjoint trajectories via local vLLM
    need_weak = not _traj_has_full_count(traj_dir / "weak_trajectories.jsonl", expected_n)
    need_meta = not _traj_has_full_count(traj_dir / "meta_trajectories.jsonl", expected_n)
    if need_meta and (out_dir / "sft_meta").exists():
        # Invalidate prior partial-slice sft_meta so Stage 2 retrains on full 600-task meta trajectories
        shutil.rmtree(out_dir / "sft_meta", ignore_errors=True)
    if need_weak or need_meta:
        start_vllm(model, Path("/content/vllm_stage1.log"))
        for need, name, teacher, mode, pdir in [
            (need_weak, "weak_trajectories.jsonl", work / "wave2_assets/weak_teacher.py", "weak", ""),
            (need_meta, "meta_trajectories.jsonl", work / "wave2_assets/meta_teacher.py", "metamorphic", str(probe_dir)),
        ]:
            if not need:
                print(f"[w2-g4] reusing bundled {name}", flush=True)
                continue
            cmd = [
                sys.executable, "-u", "-m", "meta_rsi.training.collect_trajectories",
                "--harness", str(teacher),
                "--data-root", str(work / "data/v2"),
                "--out", str(traj_dir / name),
                "--filter-mode", mode,
                "--train-offset", str(t_off),
                "--train-per-family", str(t_pf),
                "--task-backend", "openai-compat",
                "--task-model", model,
                "--task-url", "http://127.0.0.1:8000/v1",
                "--task-temp", "0.0",
                "--task-max-tokens", "2048",
                "--task-no-think",
                "--exec-timeout", "25.0",
                "--workers", "24",
            ]
            if pdir:
                cmd.extend(["--probe-dir", pdir])
            print(f"[w2-g4] Stage 1 collecting {name}...", flush=True)
            subprocess.run(cmd, check=True)
        kill_vllm()
    else:
        print("[w2-g4] Stage 1 skipped (both trajectories already pre-bundled)", flush=True)

    # Copy trajectories into out_dir so everything is packed together
    out_traj = out_dir / "trajectories"
    out_traj.mkdir(parents=True, exist_ok=True)
    for p in traj_dir.glob("*"):
        if p.is_file():
            shutil.copy2(p, out_traj / p.name)

    # Stage 2: SFT-Weak + SFT-Meta BF16 LoRA training (skip any adapter already trained on full slice)
    sft_weak_dir = out_dir / "sft_weak"
    sft_meta_dir = out_dir / "sft_meta"
    need_sft_weak = not (sft_weak_dir / "adapter_config.json").exists()
    need_sft_meta = not (sft_meta_dir / "adapter_config.json").exists()
    if need_sft_weak or need_sft_meta:
        sft_cmd = [
            sys.executable, "-u", "-m", "meta_rsi.training.split_two_timescale",
            "--mode", "sft",
            "--model", model,
            "--out-dir", str(out_dir),
            "--data-root", str(work / "data/v2"),
            "--sft-jsonl", str(traj_dir / ("meta_trajectories.jsonl" if need_sft_meta else "weak_trajectories.jsonl")),
            "--sft-lora-name", "sft_meta" if need_sft_meta else "sft_weak",
            "--sft-epochs", str(sft_epochs),
            "--sft-lr", "1e-4",
            "--sft-batch-size", "16",
            "--max-seq-len", str(max_seq_len),
            "--sync-url", "",
        ]
        if need_sft_weak and need_sft_meta:
            sft_cmd.extend(["--extra-sft", f"sft_weak={traj_dir / 'weak_trajectories.jsonl'}"])
        print(f"[w2-g4] Stage 2 starting SFT (weak={need_sft_weak}, meta={need_sft_meta})...", flush=True)
        subprocess.run(sft_cmd, check=True)
    else:
        print("[w2-g4] Stage 2 skipped (sft_weak & sft_meta already trained)", flush=True)

    # Stage 3: 4-round Alternating Online GRPO-Metamorphic (24 steps × 32 rollouts = 768 rollouts)
    from meta_rsi.harnesses.seed import SYSTEM_PROMPT as SEED_SYSTEM_PROMPT
    from meta_rsi.training.collect_trajectories import load_probes_from_dir, select_train_slice
    from meta_rsi.training.split_two_timescale import _generate_vllm_group, score_completion

    train_tasks = select_train_slice(work / "data/v2", train_offset=t_off, train_per_family=t_pf)
    probes = load_probes_from_dir(probe_dir, out_dir / "_probe_mods")
    rng = random.Random(42)
    opt_state_path = out_dir / "grpo_opt.pt"
    all_grpo_steps = []
    t_grpo_all = time.time()

    prev_lora_dir = sft_meta_dir
    prev_lora_name = "sft_meta"
    for rnd in range(1, grpo_rounds + 1):
        save_name = "grpo_meta_final" if rnd == grpo_rounds else f"grpo_r{rnd}"
        save_dir = out_dir / save_name
        # Deterministically sample this round's tasks regardless of skip so RNG stays aligned
        round_batches = [
            rng.sample(train_tasks, k=min(tasks_per_step, len(train_tasks)))
            for _ in range(steps_per_round)
        ]
        if (save_dir / "adapter_config.json").exists() and (out_dir / f"{save_name}_summary.json").exists():
            print(f"[w2-g4] Stage 3 Round {rnd}/{grpo_rounds} ({save_name}) already done, skipping", flush=True)
            rs = json.loads((out_dir / f"{save_name}_summary.json").read_text())
            all_grpo_steps.extend(rs.get("steps", []))
            prev_lora_dir = save_dir
            prev_lora_name = save_name
            continue

        print(f"\n[w2-g4] === Stage 3 GRPO Round {rnd}/{grpo_rounds} (init={prev_lora_name} -> {save_name}) ===", flush=True)
        # 3a. Sample 6*4*8 = 192 rollouts via local vLLM with --lora-modules
        start_vllm(model, Path(f"/content/vllm_grpo_r{rnd}.log"), lora_modules=[(prev_lora_name, prev_lora_dir)])
        t_gen = time.time()
        step_base = (rnd - 1) * steps_per_round
        task_jobs = []
        for si, btasks in enumerate(round_batches, 1):
            for t in btasks:
                task_jobs.append((step_base + si, t))

        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = [
                (
                    gstep,
                    t,
                    ex.submit(
                        _generate_vllm_group,
                        "http://127.0.0.1:8000/v1",
                        prev_lora_name,
                        SEED_SYSTEM_PROMPT,
                        t["prompt"],
                        n=num_generations,
                        temperature=0.8,
                        max_tokens=max_comp_tokens,
                    ),
                )
                for gstep, t in task_jobs
            ]
            flat_rollouts = []
            for gstep, t, fut in futs:
                for comp in fut.result():
                    flat_rollouts.append((gstep, t, comp))
        gen_s = round(time.time() - t_gen, 1)
        print(f"[w2-g4] Round {rnd} generated {len(flat_rollouts)} rollouts in {gen_s}s", flush=True)
        kill_vllm()

        # 3b. Score 192 rollouts in parallel on 48 CPU cores with 3 certified probes
        t_sc = time.time()
        with ThreadPoolExecutor(max_workers=24) as ex:
            scored = list(
                ex.map(
                    lambda item: score_completion(
                        item[2], item[1], work / "data/v2",
                        reward_mode="metamorphic", custom_probes=probes, timeout=25.0
                    ),
                    flat_rollouts,
                )
            )
        sc_s = round(time.time() - t_sc, 1)
        r_full = sum(1 for x in scored if x[0] >= 0.99)
        r_blk = sum(1 for x in scored if x[1] == "meta_fail_shortcut")
        r_3857 = sum(1 for x in scored if x[2])
        print(
            f"[w2-g4] Round {rnd} scored in {sc_s}s | full={r_full}/{len(scored)} "
            f"meta_blocked={r_blk} uses_3857={r_3857}",
            flush=True,
        )

        rollouts_path = out_dir / f"grpo_round_{rnd}_rollouts.jsonl"
        with open(rollouts_path, "w") as f:
            for (gstep, t, comp), (rew, cat, u3857) in zip(flat_rollouts, scored):
                f.write(
                    json.dumps(
                        {
                            "step": gstep,
                            "task_id": t["task_id"],
                            "family": t["family"],
                            "prompt": t["prompt"],
                            "completion": comp,
                            "reward": rew,
                            "cat": cat,
                            "uses_3857": u3857,
                        }
                    )
                    + "\n"
                )

        # 3c. Run 6 clipped GRPO + exact Schulman k3-KL steps against frozen sft_ref
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.training.split_two_timescale",
                "--mode", "grpo_offline_batch",
                "--model", model,
                "--out-dir", str(out_dir),
                "--sft-lora-dir", str(sft_meta_dir),
                "--init-lora-dir", str(prev_lora_dir),
                "--rollouts-jsonl", str(rollouts_path),
                "--save-lora-name", save_name,
                "--opt-state-path", str(opt_state_path),
                "--max-seq-len", str(max_seq_len),
                "--grpo-lr", "1e-5",
                "--grpo-beta", "0.02",
                "--sync-url", "",
            ],
            check=True,
        )
        rs = json.loads((out_dir / f"{save_name}_summary.json").read_text())
        all_grpo_steps.extend(rs.get("steps", []))
        prev_lora_dir = save_dir
        prev_lora_name = save_name

    # Write unified grpo_meta_final_summary.json across all 24 steps
    combined_summary = {
        "stage": "grpo_alternating",
        "lora_name": "grpo_meta_final",
        "adapter_dir": str(out_dir / "grpo_meta_final"),
        "reward_mode": "metamorphic",
        "rounds": grpo_rounds,
        "steps_per_round": steps_per_round,
        "max_steps": grpo_rounds * steps_per_round,
        "total_s": round(time.time() - t_grpo_all, 1),
        "steps": all_grpo_steps,
    }
    (out_dir / "grpo_meta_combined_summary.json").write_text(json.dumps(combined_summary, indent=2))

    # Stage 4: Final vLLM serve + ALL 9 bare-seed evaluations (eval450, real OSM, GeoAnalystBench × 3 LoRAs)
    print("\n[w2-g4] === Stage 4: Final bare-seed evaluations across 3 LoRAs ===", flush=True)
    start_vllm(
        model,
        Path("/content/vllm_stage4_eval.log"),
        lora_modules=[
            ("sft_weak", out_dir / "sft_weak"),
            ("sft_meta", out_dir / "sft_meta"),
            ("grpo_meta_final", out_dir / "grpo_meta_final"),
        ],
    )
    seed_harness = work / "meta_rsi/harnesses/seed.py"
    for lora_name, ga_tag in [
        ("sft_weak", "sft_weak_26b"),
        ("sft_meta", "sft_meta_26b"),
        ("grpo_meta_final", "grpo_meta_26b"),
    ]:
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.eval.eval_test",
                "--harness", str(seed_harness),
                "--data-root", str(work / "data/v2"),
                "--out", str(out_dir / f"eval_v2_bare_seed_{lora_name}.json"),
                "--task-backend", "openai-compat",
                "--task-url", "http://127.0.0.1:8000/v1",
                "--task-model", lora_name,
                "--task-temp", "0.0",
                "--task-max-tokens", "2048",
                "--task-no-think",
                "--max-llm-calls", "1",
                "--exec-timeout", "25.0",
                "--workers", "24",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.eval.eval_test",
                "--harness", str(seed_harness),
                "--data-root", str(work / "data/osm_eval"),
                "--out", str(out_dir / f"eval_osm_bare_seed_{lora_name}.json"),
                "--task-backend", "openai-compat",
                "--task-url", "http://127.0.0.1:8000/v1",
                "--task-model", lora_name,
                "--task-temp", "0.0",
                "--task-max-tokens", "2048",
                "--task-no-think",
                "--max-llm-calls", "1",
                "--exec-timeout", "25.0",
                "--workers", "12",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.eval.geoanalyst_eval",
                "--csv", str(work / "data/geoanalyst/GeoAnalystBench.csv"),
                "--model", lora_name,
                "--url", "http://127.0.0.1:8000/v1",
                "--out", str(out_dir / f"eval_geoanalyst_{ga_tag}.json"),
                "--workers", "8",
            ],
            check=True,
        )

    # Pack all final outputs (bf16 LoRAs + trajectories + summaries + 9 eval JSONs)
    # Cast final LoRAs to bf16 inside pack to keep archive compact (~120 MB), then split into 25 MB parts
    out_tgz = Path("/content/wave2_all_out.tgz")
    with tarfile.open(out_tgz, "w:gz", compresslevel=3) as tf:
        for p in sorted(out_dir.iterdir()):
            if p.name in ("_probe_mods", "grpo_opt.pt") or p.name.startswith("grpo_r"):
                continue
            if p.name in ("sft_weak", "sft_meta", "grpo_meta_final"):
                bf_dir = Path("/content/loras_bf16") / p.name
                tf.add(bf_dir if bf_dir.exists() else p, arcname=p.name)
            else:
                tf.add(p, arcname=p.name)
    for old_p in Path("/content").glob("wave2_out.part*"):
        old_p.unlink(missing_ok=True)
    chunk_bytes = 25 * 1024 * 1024
    part_names = []
    with open(out_tgz, "rb") as fin:
        idx = 0
        while True:
            buf = fin.read(chunk_bytes)
            if not buf:
                break
            pname = f"wave2_out.part{idx:02d}"
            (Path("/content") / pname).write_bytes(buf)
            part_names.append(pname)
            idx += 1
    Path("/content/wave2_out_manifest.json").write_text(
        json.dumps({"parts": part_names, "size_bytes": out_tgz.stat().st_size}, indent=2)
    )
    print(f"[w2-g4] ALL DONE in {(time.time()-t_start)/60:.1f} min! Packed {out_tgz} ({out_tgz.stat().st_size/1e6:.1f} MB in {len(part_names)} parts)", flush=True)
finally:
    stop_hb.set()
'''


def _run_colab(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["colab", "--auth=oauth2", *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, stdin=subprocess.DEVNULL)


def upload_chunked(session: str, local_path: Path, remote_prefix: str = "/content/wave2_bundle.part", chunk_mb: int = 25) -> None:
    chunk_bytes = chunk_mb * 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        with open(local_path, "rb") as fin:
            idx = 0
            while True:
                buf = fin.read(chunk_bytes)
                if not buf:
                    break
                p = Path(tmp) / f"part{idx:02d}"
                p.write_bytes(buf)
                _run_colab(["upload", "-s", session, str(p), f"{remote_prefix}{idx:02d}"])
                idx += 1


def download_chunked(session: str, out_tgz: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest_p = Path(tmp) / "manifest.json"
        _run_colab(["download", "-s", session, "/content/wave2_out_manifest.json", str(manifest_p)])
        manifest = json.loads(manifest_p.read_text())
        with open(out_tgz, "wb") as fout:
            for pname in manifest["parts"]:
                lp = Path(tmp) / pname
                _run_colab(["download", "-s", session, f"/content/{pname}", str(lp)])
                fout.write(lp.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-G4 96GB End-to-End Wave S4-2 Runner")
    ap.add_argument("--session", default="w2-train")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--weak-teacher", default=str(DEFAULT_WEAK_TEACHER))
    ap.add_argument("--meta-teacher", default=str(DEFAULT_META_TEACHER))
    ap.add_argument("--train-offset", type=int, default=20)
    ap.add_argument("--train-per-family", type=int, default=40)
    ap.add_argument("--sft-epochs", type=int, default=2)
    ap.add_argument("--grpo-rounds", type=int, default=4)
    ap.add_argument("--steps-per-round", type=int, default=6)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--max-completion-tokens", type=int, default=1536)
    ap.add_argument("--skip-upload", action="store_true", help="Reuse already-uploaded bundle on remote VM")
    ap.add_argument("--stop-after", action="store_true", help="Stop Colab session after download (default: KEEP ALIVE)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_tgz = out_dir / "wave2_bundle.tgz"

    if not a.skip_upload:
        info = pack_wave2_bundle(
            out_tgz=bundle_tgz,
            out_dir=out_dir,
            data_v2=REPO / "data/v2",
            data_osm=REPO / "data/osm_eval",
            geoanalyst_csv=REPO / "data/geoanalyst/GeoAnalystBench.csv",
            weak_teacher=Path(a.weak_teacher),
            meta_teacher=Path(a.meta_teacher),
            train_offset=a.train_offset,
            train_per_family=a.train_per_family,
        )
        print("[single-g4] Packed bundle:", json.dumps(info, indent=2), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        driver_f = Path(tmp) / "remote_single_g4_driver.py"
        driver_f.write_text(REMOTE_SINGLE_G4_DRIVER)

        if not a.skip_upload:
            upload_chunked(a.session, bundle_tgz)
        _run_colab(
            [
                "exec",
                "-s",
                a.session,
                "--timeout",
                "7200",
                "--env",
                f"W2_MODEL={MODEL_ID}",
                "--env",
                f"W2_TRAIN_OFFSET={a.train_offset}",
                "--env",
                f"W2_TRAIN_PER_FAMILY={a.train_per_family}",
                "--env",
                f"W2_SFT_EPOCHS={a.sft_epochs}",
                "--env",
                f"W2_GRPO_ROUNDS={a.grpo_rounds}",
                "--env",
                f"W2_STEPS_PER_ROUND={a.steps_per_round}",
                "--env",
                f"W2_MAX_SEQ_LEN={a.max_seq_len}",
                "--env",
                f"W2_MAX_COMP_TOKENS={a.max_completion_tokens}",
                "-f",
                str(driver_f),
            ]
        )

        out_tgz = out_dir / "wave2_all_out.tgz"
        download_chunked(a.session, out_tgz)
        with tarfile.open(out_tgz, "r:gz") as tf:
            tf.extractall(out_dir)

        issues = audit_downloaded_evals(out_dir)
        if issues:
            raise RuntimeError(f"Wave S4-2 evaluation audit failed: {issues}")
        print("[single-g4] ALL WAVE S4-2 TRAINING & 9 AUDITED EVALS COMPLETED CLEANLY!", flush=True)
        if a.stop_after:
            _run_colab(["stop", "-s", a.session], check=False)


if __name__ == "__main__":
    main()

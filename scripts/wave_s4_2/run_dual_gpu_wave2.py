"""One-command Dual-GPU Split Launcher for Sprint 4 Wave S4-2 (GSB v2 Two-Timescale RSI).

Architecture (zero-NCCL Dual-GPU Split, verified in v5 on 2× Colab G4 96 GB):
  - `w2-infer`: serves `google/gemma-4-26B-A4B-it` via vLLM (`--enable-lora`) + `lora_sync` sidecar.
  - `w2-train`: loads 1× bf16 base copy + dual PEFT adapters (`sft_ref` + `policy`), runs:
      1. `SFT-Weak` control LoRA (`sft_weak`)
      2. `SFT-Meta` warm-start LoRA (`sft_meta`)
      3. `GRPO-Metamorphic` (`24` online steps on `data/v2` `train[20:60]` with 3 certified probes)
  - Local workstation bridges ports `8000/8003` (`w2-infer`) <-> `8001/8003` (`w2-train`) via
    auto-reconnecting SSH ProxyCommand tunnels.
  - Post-training bare-`seed.py` evaluations run directly on `w2-infer` (48 CPU cores, zero tunnel RTT)
    across `data/v2` (`eval450`), `data/osm_eval` (`n=24`), and `GeoAnalystBench` (`n=17`).
  - Downloads all LoRA checkpoints, training summaries, and audited eval JSONs to `runs/v8_s4_wave2_v2/`,
    and ALWAYS stops both Colab sessions on exit (unless `--keep`).

Usage:
    # Dry-run bundle packing & integrity check (zero GPU usage):
    nix-shell --run "python -u scripts/wave_s4_2/run_dual_gpu_wave2.py --pack-only"

    # Full run once Colab G4 / H100 / A100 recovers:
    nix-shell --run "python -u scripts/wave_s4_2/run_dual_gpu_wave2.py --gpu G4"
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.wave_s4_2.collect_v2_trajectories import (
    DEFAULT_META_TEACHER,
    DEFAULT_OUT_DIR,
    DEFAULT_WEAK_TEACHER,
    is_clean_trajectories,
    stage_certified_probes,
)

MODEL_ID = "google/gemma-4-26B-A4B-it"


def pack_wave2_bundle(
    *,
    out_tgz: Path,
    out_dir: Path,
    data_v2: Path,
    data_osm: Path,
    geoanalyst_csv: Path,
    weak_teacher: Path,
    meta_teacher: Path,
    train_offset: int,
    train_per_family: int,
) -> dict:
    """Pack minimal code, disjoint GSB v2 train[20:60] + test 450, OSM, GeoAnalystBench, probes & trajectories."""
    probe_dir = stage_certified_probes(out_dir)

    rows = [json.loads(ln) for ln in open(data_v2 / "tasks.jsonl")]
    by_fam_train: dict[str, list[dict]] = defaultdict(list)
    test_rows: list[dict] = []
    for r in rows:
        if r["split"] == "train":
            by_fam_train[r["family"]].append(r)
        elif r["split"] == "test":
            test_rows.append(r)

    train_selected: list[dict] = []
    for fam in sorted(by_fam_train):
        train_selected.extend(by_fam_train[fam][train_offset : train_offset + train_per_family])
    selected = train_selected + test_rows

    traj_dir = out_dir / "trajectories"
    expected_total = len(train_selected)
    has_weak_traj = is_clean_trajectories(traj_dir / "weak_trajectories.jsonl", expected_total)
    has_meta_traj = is_clean_trajectories(traj_dir / "meta_trajectories.jsonl", expected_total)

    out_tgz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tgz, "w:gz", compresslevel=3) as tf:
        # 1. Core Python packages
        for pkg in ("benchmarks", "meta_rsi"):
            for p in sorted((REPO / pkg).rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                tf.add(p, arcname=str(p.relative_to(REPO)))

        # 2. Filtered GSB v2 tasks.jsonl + variant directories
        with tempfile.TemporaryDirectory() as tmp:
            tfile = Path(tmp) / "tasks.jsonl"
            tfile.write_text("".join(json.dumps(t) + "\n" for t in selected))
            tf.add(tfile, arcname="data/v2/tasks.jsonl")
        for t in selected:
            tdir = data_v2 / t["task_id"]
            if tdir.exists():
                tf.add(tdir, arcname=f"data/v2/{t['task_id']}")

        # 3. Real 4-city OSM slice
        for p in sorted(data_osm.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(REPO)))

        # 4. External GeoAnalystBench CSV
        tf.add(geoanalyst_csv, arcname=str(geoanalyst_csv.relative_to(REPO)))

        # 5. Certified probes & teacher harnesses
        for p in sorted(probe_dir.glob("*.py")):
            tf.add(p, arcname=f"wave2_assets/certified_probes/{p.name}")
        tf.add(weak_teacher, arcname="wave2_assets/weak_teacher.py")
        tf.add(meta_teacher, arcname="wave2_assets/meta_teacher.py")

        # 6. Pre-collected trajectories (if already collected cleanly via Vertex AI)
        if has_weak_traj:
            tf.add(traj_dir / "weak_trajectories.jsonl", arcname="wave2_assets/trajectories/weak_trajectories.jsonl")
            tf.add(
                traj_dir / "weak_trajectories.summary.json",
                arcname="wave2_assets/trajectories/weak_trajectories.summary.json",
            )
        if has_meta_traj:
            tf.add(traj_dir / "meta_trajectories.jsonl", arcname="wave2_assets/trajectories/meta_trajectories.jsonl")
            tf.add(
                traj_dir / "meta_trajectories.summary.json",
                arcname="wave2_assets/trajectories/meta_trajectories.summary.json",
            )

    return {
        "bundle": str(out_tgz),
        "size_mb": round(out_tgz.stat().st_size / (1024 * 1024), 2),
        "n_train": len(train_selected),
        "n_test": len(test_rows),
        "has_precollected_weak": has_weak_traj,
        "has_precollected_meta": has_meta_traj,
    }


REMOTE_INFER_SETUP = r'''
import json, os, subprocess, sys, tarfile, threading, time, urllib.request
from pathlib import Path

stop_hb = threading.Event()
def _hb():
    t0 = time.time()
    while not stop_hb.wait(15.0):
        print(f"[w2-infer hb] elapsed={time.time()-t0:.0f}s", flush=True)
threading.Thread(target=_hb, daemon=True).start()

try:
    work = Path("/content/wave2_work")
    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/content/wave2_bundle.tgz", "r:gz") as tf:
        tf.extractall(work)
    os.chdir(work)
    sys.path.insert(0, str(work))

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)

    needed = []
    for mod, pkg in [
        ("geopandas", "geopandas"),
        ("pyproj", "pyproj"),
        ("shapely", "shapely"),
        ("rasterio", "rasterio"),
        ("scipy", "scipy"),
        ("vllm", "vllm"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        print("[w2-infer] installing:", needed, flush=True)
        try:
            subprocess.run(["uv", "pip", "install", "--system", *needed], check=True)
        except FileNotFoundError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)

    def http_ok(url):
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    model = os.environ.get("W2_MODEL", "google/gemma-4-26B-A4B-it")
    gpu_mem = os.environ.get("W2_VLLM_GPU_MEM", "0.88")

    if not http_ok("http://127.0.0.1:8000/v1/models"):
        subprocess.run(["pkill", "-9", "-f", "vllm|VLLM|EngineCore"], check=False)
        time.sleep(2)
        env = os.environ.copy()
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        log_f = open("/content/vllm_infer.log", "w")
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
        print("[w2-infer] launching vLLM:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        t0 = time.time()
        while not http_ok("http://127.0.0.1:8000/v1/models"):
            if proc.poll() is not None:
                print(Path("/content/vllm_infer.log").read_text()[-3000:], file=sys.stderr)
                raise RuntimeError(f"vLLM exited with code {proc.returncode}")
            if time.time() - t0 > 600:
                raise TimeoutError("Timed out waiting for vLLM")
            time.sleep(5)
        print(f"[w2-infer] vLLM READY in {time.time()-t0:.1f}s", flush=True)
    else:
        print("[w2-infer] reusing live vLLM on :8000", flush=True)

    if not http_ok("http://127.0.0.1:8003/health"):
        sync_log = open("/content/lora_sync.log", "w")
        subprocess.Popen(
            [
                sys.executable, "-m", "meta_rsi.training.lora_sync",
                "--host", "127.0.0.1", "--port", "8003",
                "--vllm-url", "http://127.0.0.1:8000",
                "--lora-root", "/content/loras",
            ],
            stdout=sync_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = time.time()
        while not http_ok("http://127.0.0.1:8003/health"):
            if time.time() - t0 > 30:
                raise TimeoutError("lora_sync failed to start")
            time.sleep(1)
        print("[w2-infer] lora_sync READY on :8003", flush=True)

    # Collect trajectories on w2-infer if not already pre-collected via Vertex AI
    # Note: bundled data/v2/tasks.jsonl is pre-sliced to train[train_offset : train_offset+train_per_family],
    # so remote train_offset inside the bundle is always 0.
    traj_dir = work / "wave2_assets" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    t_pf = os.environ.get("W2_TRAIN_PER_FAMILY", "40")
    t_off = "0"

    for name, teacher, mode, pdir in [
        ("weak_trajectories.jsonl", work / "wave2_assets/weak_teacher.py", "weak", ""),
        ("meta_trajectories.jsonl", work / "wave2_assets/meta_teacher.py", "metamorphic", str(work / "wave2_assets/certified_probes")),
    ]:
        out_p = traj_dir / name
        if out_p.exists() and out_p.with_suffix(".summary.json").exists():
            print(f"[w2-infer] reusing pre-collected {name}", flush=True)
            continue
        cmd = [
            sys.executable, "-u", "-m", "meta_rsi.training.collect_trajectories",
            "--harness", str(teacher),
            "--data-root", str(work / "data/v2"),
            "--out", str(out_p),
            "--filter-mode", mode,
            "--train-offset", t_off,
            "--train-per-family", t_pf,
            "--task-backend", "openai-compat",
            "--task-model", model,
            "--task-url", "http://127.0.0.1:8000/v1",
            "--task-temp", "0.0",
            "--task-max-tokens", "2048",
            "--task-no-think",
            "--exec-timeout", "25.0",
            "--workers", "16",
        ]
        if pdir:
            cmd.extend(["--probe-dir", pdir])
        print(f"[w2-infer] collecting {name} locally against vLLM...", flush=True)
        subprocess.run(cmd, check=True)

    # Pack trajectories if generated remotely so host can copy them to w2-train
    with tarfile.open("/content/wave2_trajs.tgz", "w:gz") as tf:
        for p in sorted(traj_dir.glob("*")):
            if p.is_file():
                tf.add(p, arcname=p.name)
    print("[w2-infer] setup & trajectory staging complete.", flush=True)
finally:
    stop_hb.set()
'''

REMOTE_TRAIN_RUN = r'''
import os, subprocess, sys, tarfile, threading, time, urllib.request
from pathlib import Path

stop_hb = threading.Event()
def _hb():
    t0 = time.time()
    while not stop_hb.wait(15.0):
        print(f"[w2-train-setup hb] elapsed={time.time()-t0:.0f}s", flush=True)
threading.Thread(target=_hb, daemon=True).start()

try:
    work = Path("/content/wave2_work")
    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/content/wave2_bundle.tgz", "r:gz") as tf:
        tf.extractall(work)
    if Path("/content/wave2_trajs.tgz").exists():
        tdir = work / "wave2_assets" / "trajectories"
        tdir.mkdir(parents=True, exist_ok=True)
        with tarfile.open("/content/wave2_trajs.tgz", "r:gz") as tf:
            tf.extractall(tdir)
    os.chdir(work)
    sys.path.insert(0, str(work))

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)

    # Remove torchao to avoid peft import conflict on Colab
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

    needed = []
    for mod, pkg in [
        ("peft", "peft"),
        ("safetensors", "safetensors"),
        ("accelerate", "accelerate"),
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
        print("[w2-train] installing:", needed, flush=True)
        try:
            subprocess.run(["uv", "pip", "install", "--system", *needed], check=True)
        except FileNotFoundError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)

    # Verify reverse SSH tunnels to w2-infer (vLLM :8001, lora_sync :8003)
    for url in ("http://127.0.0.1:8001/v1/models", "http://127.0.0.1:8003/health"):
        for attempt in range(20):
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    assert r.status == 200
                print(f"[w2-train] tunnel OK: {url}", flush=True)
                break
            except Exception as exc:
                if attempt == 19:
                    raise RuntimeError(f"Tunnel check failed for {url}: {exc}")
                time.sleep(2)

    stop_hb.set()
    model = os.environ.get("W2_MODEL", "google/gemma-4-26B-A4B-it")
    out_dir = Path("/content/wave2_train_out")
    traj_dir = work / "wave2_assets" / "trajectories"
    cmd = [
        sys.executable, "-u", "-m", "meta_rsi.training.split_two_timescale",
        "--mode", "sft_and_grpo",
        "--model", model,
        "--out-dir", str(out_dir),
        "--data-root", str(work / "data/v2"),
        "--extra-sft", f"sft_weak={traj_dir / 'weak_trajectories.jsonl'}",
        "--sft-jsonl", str(traj_dir / "meta_trajectories.jsonl"),
        "--sft-lora-name", "sft_meta",
        "--sft-epochs", os.environ.get("W2_SFT_EPOCHS", "2"),
        "--sft-lr", os.environ.get("W2_SFT_LR", "1e-4"),
        "--sft-batch-size", os.environ.get("W2_SFT_BS", "16"),
        "--grpo-prefix", "grpo_meta",
        "--reward-mode", "metamorphic",
        "--grpo-steps", os.environ.get("W2_GRPO_STEPS", "24"),
        "--grpo-lr", os.environ.get("W2_GRPO_LR", "1e-5"),
        "--grpo-beta", os.environ.get("W2_GRPO_BETA", "0.02"),
        "--tasks-per-step", "4",
        "--num-generations", "8",
        "--sync-every", "2",
        "--train-offset", "0",
        "--train-per-family", os.environ.get("W2_TRAIN_PER_FAMILY", "40"),
        "--max-seq-len", os.environ.get("W2_MAX_SEQ_LEN", "2560"),
        "--max-completion-tokens", os.environ.get("W2_MAX_COMP_TOKENS", "1536"),
        "--exec-timeout", "25.0",
        "--probe-dir", str(work / "wave2_assets/certified_probes"),
        "--vllm-url", "http://127.0.0.1:8001/v1",
        "--sync-url", "http://127.0.0.1:8003",
    ]
    print("[w2-train] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    out_tgz = Path("/content/wave2_train_out.tgz")
    with tarfile.open(out_tgz, "w:gz") as tf:
        for p in sorted(out_dir.iterdir()):
            if p.name in ("latest_policy", "_probe_mods"):
                continue
            tf.add(p, arcname=p.name)
    print(f"[w2-train] packed {out_tgz} ({out_tgz.stat().st_size/1e6:.1f} MB)", flush=True)
finally:
    stop_hb.set()
'''

REMOTE_INFER_EVAL = r'''
import json, os, subprocess, sys, tarfile, threading, time, urllib.request
from pathlib import Path

work = Path("/content/wave2_work")
os.chdir(work)
sys.path.insert(0, str(work))

stop_hb = threading.Event()
def _hb():
    t0 = time.time()
    while not stop_hb.wait(15.0):
        print(f"[w2-eval hb] elapsed={time.time()-t0:.0f}s", flush=True)
threading.Thread(target=_hb, daemon=True).start()

def ensure_lora_loaded(name: str):
    lora_dir = Path("/content/loras") / name
    assert lora_dir.exists(), f"Missing synced LoRA on w2-infer: {lora_dir}"
    body = json.dumps({"lora_name": name, "lora_path": str(lora_dir)}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/load_lora_adapter",
        body,
        {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"[w2-eval] loaded LoRA {name}: HTTP {r.status}", flush=True)
    except Exception as exc:
        print(f"[w2-eval] load_lora_adapter({name}) note: {exc}", flush=True)

eval_out = Path("/content/wave2_eval_out")
eval_out.mkdir(parents=True, exist_ok=True)
seed_harness = work / "meta_rsi/harnesses/seed.py"

try:
    for lora_name, ga_tag in [
        ("sft_weak", "sft_weak_26b"),
        ("sft_meta", "sft_meta_26b"),
        ("grpo_meta_final", "grpo_meta_26b"),
    ]:
        ensure_lora_loaded(lora_name)
        # 1. GSB v2 eval450 (bare seed.py)
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.eval.eval_test",
                "--harness", str(seed_harness),
                "--data-root", str(work / "data/v2"),
                "--out", str(eval_out / f"eval_v2_bare_seed_{lora_name}.json"),
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
        # 2. Real 4-city OSM (n=24, bare seed.py)
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.eval.eval_test",
                "--harness", str(seed_harness),
                "--data-root", str(work / "data/osm_eval"),
                "--out", str(eval_out / f"eval_osm_bare_seed_{lora_name}.json"),
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
        # 3. External GeoAnalystBench deterministic audit (n=17)
        subprocess.run(
            [
                sys.executable, "-u", "-m", "meta_rsi.eval.geoanalyst_eval",
                "--csv", str(work / "data/geoanalyst/GeoAnalystBench.csv"),
                "--model", lora_name,
                "--url", "http://127.0.0.1:8000/v1",
                "--out", str(eval_out / f"eval_geoanalyst_{ga_tag}.json"),
                "--workers", "8",
            ],
            check=True,
        )

    out_tgz = Path("/content/wave2_eval_out.tgz")
    with tarfile.open(out_tgz, "w:gz") as tf:
        for p in sorted(eval_out.glob("*.json")):
            tf.add(p, arcname=p.name)
    print(f"[w2-eval] packed {out_tgz} cleanly.", flush=True)
finally:
    stop_hb.set()
'''


def _run_colab(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["colab", "--auth=oauth2", *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, stdin=subprocess.DEVNULL)


class PersistentTunnel:
    """Auto-reconnecting SSH tunnel over `colab ssh --proxy-mode` (handles 60-min WebSocket drops)."""

    def __init__(self, session: str, forwards: list[str]) -> None:
        self.session = session
        self.forwards = forwards
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        cmd = [
            "ssh",
            "-o",
            f"ProxyCommand=colab --auth=oauth2 ssh --proxy-mode -s {self.session}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=25",
            "-o",
            "ServerAliveCountMax=3",
            "-N",
            *self.forwards,
            "root@colab",
        ]
        while not self._stop.is_set():
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            self._proc.wait()
            if not self._stop.is_set():
                time.sleep(3)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def wait_local_http(url: str, timeout_s: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for tunneled endpoint {url}")


def audit_downloaded_evals(out_dir: Path) -> list[str]:
    issues = []
    for p in sorted(out_dir.glob("eval_*.json")):
        d = json.loads(p.read_text())
        if "ALL" in d:
            n = int(d["ALL"]["n"])
            u = d.get("usage") or {}
        else:
            s = d.get("summary") or {}
            n = int(s.get("n", 0))
            u = s.get("usage") or {}
        uerr = int(u.get("errors", 1))
        calls = int(u.get("calls", 0))
        if uerr != 0 or calls < n:
            issues.append(f"{p.name}: uerr={uerr}, calls={calls}<n={n}")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-GPU Split Launcher for Wave S4-2 (GSB v2 Two-Timescale RSI)")
    ap.add_argument("--gpu", default="G4", choices=["G4", "H100", "A100"])
    ap.add_argument("--infer-session", default="w2-infer")
    ap.add_argument("--train-session", default="w2-train")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--weak-teacher", default=str(DEFAULT_WEAK_TEACHER))
    ap.add_argument("--meta-teacher", default=str(DEFAULT_META_TEACHER))
    ap.add_argument("--train-offset", type=int, default=20)
    ap.add_argument("--train-per-family", type=int, default=40)
    ap.add_argument("--grpo-steps", type=int, default=24)
    ap.add_argument("--sft-epochs", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--max-completion-tokens", type=int, default=1536)
    ap.add_argument("--pack-only", action="store_true", help="Pack & verify wave2_bundle.tgz without provisioning GPUs")
    ap.add_argument("--reuse-sessions", action="store_true")
    ap.add_argument("--keep", action="store_true", help="Keep Colab sessions running after download")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_tgz = out_dir / "wave2_bundle.tgz"

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
    print("[wave2] Packed bundle:", json.dumps(info, indent=2), flush=True)
    if a.pack_only:
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        f_infer_setup = tmp_p / "remote_infer_setup.py"
        f_train_run = tmp_p / "remote_train_run.py"
        f_infer_eval = tmp_p / "remote_infer_eval.py"
        f_infer_setup.write_text(REMOTE_INFER_SETUP)
        f_train_run.write_text(REMOTE_TRAIN_RUN)
        f_infer_eval.write_text(REMOTE_INFER_EVAL)

        t_infer: PersistentTunnel | None = None
        t_train: PersistentTunnel | None = None
        try:
            if not a.reuse_sessions:
                _run_colab(["new", "-s", a.infer_session, "--gpu", a.gpu])
                _run_colab(["new", "-s", a.train_session, "--gpu", a.gpu])

            # 1. Upload bundle to both sessions
            _run_colab(["upload", "-s", a.infer_session, str(bundle_tgz), "/content/wave2_bundle.tgz"])
            _run_colab(["upload", "-s", a.train_session, str(bundle_tgz), "/content/wave2_bundle.tgz"])

            # 2. Boot vLLM + lora_sync (+ collect trajectories if not pre-collected) on w2-infer
            gpu_mem = "0.88" if a.gpu == "G4" else "0.90"
            _run_colab(
                [
                    "exec", "-s", a.infer_session, "--timeout", "1800",
                    "--env", f"W2_MODEL={MODEL_ID}",
                    "--env", f"W2_VLLM_GPU_MEM={gpu_mem}",
                    "--env", f"W2_TRAIN_OFFSET={a.train_offset}",
                    "--env", f"W2_TRAIN_PER_FAMILY={a.train_per_family}",
                    "-f", str(f_infer_setup),
                ]
            )

            # Mirror remote-collected trajectories to local & w2-train
            local_trajs_tgz = out_dir / "wave2_trajs.tgz"
            _run_colab(["download", "-s", a.infer_session, "/content/wave2_trajs.tgz", str(local_trajs_tgz)])
            traj_dir = out_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local_trajs_tgz, "r:gz") as tf:
                tf.extractall(traj_dir)
            _run_colab(["upload", "-s", a.train_session, str(local_trajs_tgz), "/content/wave2_trajs.tgz"])

            # 3. Establish persistent SSH tunnel bridge (w2-infer :8000/:8003 -> host :18001/:18003 -> w2-train :8001/:8003)
            t_infer = PersistentTunnel(
                a.infer_session,
                ["-L", "127.0.0.1:18001:127.0.0.1:8000", "-L", "127.0.0.1:18003:127.0.0.1:8003"],
            )
            t_infer.start()
            wait_local_http("http://127.0.0.1:18001/v1/models", timeout_s=60.0)
            wait_local_http("http://127.0.0.1:18003/health", timeout_s=30.0)

            t_train = PersistentTunnel(
                a.train_session,
                ["-R", "127.0.0.1:8001:127.0.0.1:18001", "-R", "127.0.0.1:8003:127.0.0.1:18003"],
            )
            t_train.start()
            time.sleep(5)

            # 4. Run Dual-GPU Split SFT-Weak + SFT-Meta + GRPO-Metamorphic on w2-train
            _run_colab(
                [
                    "exec", "-s", a.train_session, "--timeout", "3600",
                    "--env", f"W2_MODEL={MODEL_ID}",
                    "--env", f"W2_SFT_EPOCHS={a.sft_epochs}",
                    "--env", f"W2_GRPO_STEPS={a.grpo_steps}",
                    "--env", f"W2_TRAIN_OFFSET={a.train_offset}",
                    "--env", f"W2_TRAIN_PER_FAMILY={a.train_per_family}",
                    "--env", f"W2_MAX_SEQ_LEN={a.max_seq_len}",
                    "--env", f"W2_MAX_COMP_TOKENS={a.max_completion_tokens}",
                    "-f", str(f_train_run),
                ]
            )

            # 5. Download trained LoRAs & summaries immediately, then release w2-train early to save CU!
            train_tgz = out_dir / "wave2_train_out.tgz"
            _run_colab(["download", "-s", a.train_session, "/content/wave2_train_out.tgz", str(train_tgz)])
            with tarfile.open(train_tgz, "r:gz") as tf:
                tf.extractall(out_dir)
            if t_train is not None:
                t_train.stop()
            if not a.keep:
                _run_colab(["stop", "-s", a.train_session], check=False)

            # 6. Run all 9 bare-seed evaluations (eval450 + OSM + GeoAnalystBench × 3 LoRAs) directly on w2-infer
            _run_colab(["exec", "-s", a.infer_session, "--timeout", "3600", "-f", str(f_infer_eval)])
            eval_tgz = out_dir / "wave2_eval_out.tgz"
            _run_colab(["download", "-s", a.infer_session, "/content/wave2_eval_out.tgz", str(eval_tgz)])
            with tarfile.open(eval_tgz, "r:gz") as tf:
                tf.extractall(out_dir)

            issues = audit_downloaded_evals(out_dir)
            if issues:
                raise RuntimeError(f"Wave S4-2 evaluation audit failed: {issues}")
            print("[wave2] ALL WAVE S4-2 TRAINING & AUDITED EVALS COMPLETED CLEANLY!", flush=True)
        finally:
            if t_train is not None:
                t_train.stop()
            if t_infer is not None:
                t_infer.stop()
            if not a.keep:
                _run_colab(["stop", "-s", a.train_session], check=False)
                _run_colab(["stop", "-s", a.infer_session], check=False)


if __name__ == "__main__":
    main()

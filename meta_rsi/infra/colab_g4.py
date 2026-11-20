"""One-command Colab G4 launcher for meta-rsi Harness Evolution.

Packages only `benchmarks/`, `meta_rsi/`, and the exact task subset needed for the run
(a few MB instead of the full 2,700-task dataset), provisions a Colab G4 VM,
starts a local `vllm serve` OpenAI-compatible server for the frozen Task LLM,
runs `meta_rsi.evolve` (using `gemini-rest` for the low-frequency Meta-Agent),
downloads `runs/<exp>` back locally, and stops the VM cleanly unless `--keep` is set.

Usage:
    GEMINI_API_KEY=... python -m meta_rsi.infra.colab_g4 --exp g4_e4b_meta --gate metamorphic --generations 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ..evaluator import load_task_splits


def pack_bundle(
    repo_root: Path,
    data_root: Path,
    out_tgz: Path,
    *,
    dev_per_family: int,
    test_seen_per_family: int,
    test_heldout_per_family: int,
) -> int:
    """Create a compact tarball containing benchmarks/, meta_rsi/, and only the selected task dirs."""
    splits = load_task_splits(
        data_root,
        dev_per_family=dev_per_family,
        test_seen_per_family=test_seen_per_family,
        test_heldout_per_family=test_heldout_per_family,
    )
    selected = splits["dev"] + splits["test_seen"] + splits["test_heldout"]
    seen_ids = set()
    unique_tasks = []
    for t in selected:
        if t["task_id"] not in seen_ids:
            seen_ids.add(t["task_id"])
            unique_tasks.append(t)

    with tarfile.open(out_tgz, "w:gz") as tf:
        for pkg in ("benchmarks", "meta_rsi"):
            for p in sorted((repo_root / pkg).rglob("*.py")):
                tf.add(p, arcname=str(p.relative_to(repo_root)))
        with tempfile.TemporaryDirectory() as tmp:
            tfile = Path(tmp) / "tasks.jsonl"
            tfile.write_text("".join(json.dumps(t) + "\n" for t in unique_tasks))
            tf.add(tfile, arcname="data/v1/tasks.jsonl")
        for t in unique_tasks:
            tdir = data_root / t["task_id"]
            tf.add(tdir, arcname=f"data/v1/{t['task_id']}")
    return len(unique_tasks)


REMOTE_DRIVER = r'''
import os, subprocess, sys, time, urllib.request, tarfile
from pathlib import Path

work = Path("/content/rsi_work")
work.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/rsi_bundle.tgz", "r:gz") as tf:
    tf.extractall(work)

os.chdir(work)
sys.path.insert(0, str(work))

# Ensure geospatial + vLLM dependencies are present.
def ensure_pkgs():
    needed = []
    for mod, pkg in [("geopandas", "geopandas"), ("pyproj", "pyproj"), ("shapely", "shapely"), ("vllm", "vllm")]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        print("[remote] installing:", needed, flush=True)
        subprocess.run(["uv", "pip", "install", "--system", *needed], check=True)

ensure_pkgs()

model = os.environ["RSI_TASK_MODEL"]
port = int(os.environ.get("RSI_VLLM_PORT", "8000"))
health_url = f"http://127.0.0.1:{port}/v1/models"

def server_ready():
    try:
        with urllib.request.urlopen(health_url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

vllm_proc = None
if not server_ready():
    log_f = open("/content/vllm.log", "w")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", os.environ.get("RSI_GPU_MEM", "0.85"),
        "--max-model-len", os.environ.get("RSI_MAX_LEN", "8192"),
        "--max-num-seqs", os.environ.get("RSI_MAX_SEQS", "64"),
    ]
    print("[remote] starting vLLM:", " ".join(cmd), flush=True)
    vllm_proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    t0 = time.time()
    while not server_ready():
        if vllm_proc.poll() is not None:
            print(Path("/content/vllm.log").read_text()[-2500:], file=sys.stderr)
            raise RuntimeError(f"vLLM exited early with code {vllm_proc.returncode}")
        if time.time() - t0 > 480:
            raise TimeoutError("Timed out waiting for vLLM server")
        time.sleep(5)
    print(f"[remote] vLLM ready in {time.time() - t0:.1f}s", flush=True)
else:
    print("[remote] reusing live vLLM server on port", port, flush=True)

exp = os.environ["RSI_EXP"]
evolve_cmd = [
    sys.executable, "-m", "meta_rsi.evolve",
    "--exp", exp,
    "--runs-dir", "/content/rsi_work/runs",
    "--data-root", "/content/rsi_work/data/v1",
    "--gate", os.environ["RSI_GATE"],
    "--generations", os.environ["RSI_GENS"],
    "--dev-per-family", os.environ["RSI_DEV_PF"],
    "--test-seen-per-family", os.environ["RSI_SEEN_PF"],
    "--test-heldout-per-family", os.environ["RSI_HO_PF"],
    "--workers", os.environ.get("RSI_WORKERS", "12"),
    "--task-backend", "openai-compat",
    "--task-url", f"http://127.0.0.1:{port}/v1",
    "--task-model", model,
    "--meta-backend", "gemini-rest",
    "--meta-model", os.environ["RSI_META_MODEL"],
]
if os.environ.get("RSI_TASK_NO_THINK") == "1":
    evolve_cmd.append("--task-no-think")
if os.environ.get("RSI_EVAL_ORACLE_ALL") == "1":
    evolve_cmd.append("--eval-oracle-all")

print("[remote] running:", " ".join(evolve_cmd), flush=True)
subprocess.run(evolve_cmd, check=True)

out_tgz = Path(f"/content/{exp}_results.tgz")
with tarfile.open(out_tgz, "w:gz") as tf:
    tf.add(work / "runs" / exp, arcname=exp)
print("[remote] packed results:", out_tgz, flush=True)
'''


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RSI Harness Evolution on a Colab G4 GPU via colab CLI")
    ap.add_argument("--session", default="rsi-g4")
    ap.add_argument("--gpu", default="G4")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gate", default="metamorphic", choices=["naive", "preserve", "metamorphic"])
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--dev-per-family", type=int, default=1)
    ap.add_argument("--test-seen-per-family", type=int, default=1)
    ap.add_argument("--test-heldout-per-family", type=int, default=6)
    ap.add_argument("--task-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--meta-model", default="gemini-4-argon")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--task-no-think", action="store_true")
    ap.add_argument("--eval-oracle-all", action="store_true")
    ap.add_argument("--reuse-session", action="store_true", help="Reuse an already running Colab session")
    ap.add_argument("--keep", action="store_true", help="Do NOT stop the Colab VM after downloading results")
    a = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("Error: set GEMINI_API_KEY in environment so the Colab VM can call Gemini REST for the Meta-Agent.", file=sys.stderr)
        sys.exit(2)

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "v1"

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "rsi_bundle.tgz"
        driver = Path(tmp) / "remote_driver.py"
        n_tasks = pack_bundle(
            repo_root,
            data_root,
            bundle,
            dev_per_family=a.dev_per_family,
            test_seen_per_family=a.test_seen_per_family,
            test_heldout_per_family=a.test_heldout_per_family,
        )
        mb = bundle.stat().st_size / (1024 * 1024)
        print(f"Packed {n_tasks} tasks + code into {bundle.name} ({mb:.2f} MB)", flush=True)
        driver.write_text(REMOTE_DRIVER)

        if not a.reuse_session:
            _run(["colab", "new", "-s", a.session, "--gpu", a.gpu])
        try:
            _run(["colab", "upload", "-s", a.session, str(bundle), "/content/rsi_bundle.tgz"])
            env_flags = [
                "--env", f"GEMINI_API_KEY={api_key}",
                "--env", f"RSI_EXP={a.exp}",
                "--env", f"RSI_GATE={a.gate}",
                "--env", f"RSI_GENS={a.generations}",
                "--env", f"RSI_DEV_PF={a.dev_per_family}",
                "--env", f"RSI_SEEN_PF={a.test_seen_per_family}",
                "--env", f"RSI_HO_PF={a.test_heldout_per_family}",
                "--env", f"RSI_TASK_MODEL={a.task_model}",
                "--env", f"RSI_META_MODEL={a.meta_model}",
                "--env", f"RSI_WORKERS={a.workers}",
                "--env", f"RSI_TASK_NO_THINK={'1' if a.task_no_think else '0'}",
                "--env", f"RSI_EVAL_ORACLE_ALL={'1' if a.eval_oracle_all else '0'}",
            ]
            # Avoid printing GEMINI_API_KEY to stdout.
            exec_cmd = ["colab", "exec", "-s", a.session, *env_flags, "-f", str(driver)]
            print(f"$ colab exec -s {a.session} [env redacted] -f remote_driver.py", flush=True)
            subprocess.run(exec_cmd, check=True)

            out_dir = repo_root / "runs"
            out_dir.mkdir(parents=True, exist_ok=True)
            local_tgz = out_dir / f"{a.exp}_results.tgz"
            _run(["colab", "download", "-s", a.session, f"/content/{a.exp}_results.tgz", str(local_tgz)])
            with tarfile.open(local_tgz, "r:gz") as tf:
                tf.extractall(out_dir)
            local_tgz.unlink(missing_ok=True)
            print(f"Downloaded and extracted results to {out_dir / a.exp}", flush=True)
        finally:
            if not a.keep:
                _run(["colab", "stop", "-s", a.session], check=False)


if __name__ == "__main__":
    main()

"""Remote Colab G4 helper: install vLLM if missing and start OpenAI-compatible server."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-E4B-it")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
MAX_LEN = os.environ.get("VLLM_MAX_LEN", "8192")
GPU_MEM = os.environ.get("VLLM_GPU_MEM", "0.85")
HEALTH_URL = f"http://127.0.0.1:{PORT}/v1/models"


def is_ready() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    try:
        import vllm  # noqa: F401
        print("[g4] vllm already installed:", vllm.__version__, flush=True)
    except ImportError:
        print("[g4] installing vllm...", flush=True)
        installer = ["uv", "pip", "install", "--system", "vllm"]
        try:
            subprocess.run(installer, check=True)
        except FileNotFoundError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

    if is_ready():
        print(f"[g4] vLLM already serving on port {PORT}", flush=True)
        return

    log_path = Path("/content/vllm_server.log")
    log_f = open(log_path, "w")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        GPU_MEM,
        "--max-model-len",
        MAX_LEN,
        "--max-num-seqs",
        os.environ.get("VLLM_MAX_SEQS", "128"),
    ]
    print("[g4] launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while not is_ready():
        if proc.poll() is not None:
            print(log_path.read_text()[-3000:], file=sys.stderr)
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        if time.time() - t0 > 600:
            raise TimeoutError("Timed out waiting for vLLM server")
        if int(time.time() - t0) % 20 == 0:
            print(f"[g4] waiting for vLLM... ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(5)
    print(f"[g4] vLLM READY in {time.time() - t0:.1f}s (pid={proc.pid})", flush=True)


if __name__ == "__main__":
    main()

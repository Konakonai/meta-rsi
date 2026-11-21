"""Lightweight HTTP LoRA adapter sync server & client for Dual-G4 Split vLLM training.

Server side (runs alongside vLLM on the inference GPU):
    python -m meta_rsi.training.lora_sync --port 8003 --vllm-url http://127.0.0.1:8000 --lora-root /content/loras

Client side (called from the trainer GPU after saving a PEFT adapter):
    push_lora_adapter(adapter_dir, sync_url="http://127.0.0.1:8003", lora_name="grpo_s01", unload_prev="grpo_s00")
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _validate_name(name: str) -> str:
    if not name or not _SAFE_NAME_RE.match(name):
        raise ValueError(f"Invalid LoRA adapter name: {name!r}")
    return name


def _post_json(url: str, payload: dict, timeout: float = 60.0) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def pack_adapter_tar_gz(adapter_dir: Path, *, cast_bf16: bool = True) -> bytes:
    """Pack a PEFT adapter directory into gzip-compressed tar bytes.

    When `cast_bf16=True` and `safetensors`/`torch` are available, downcasts fp32 LoRA weights
    to bfloat16 in a temporary copy (cutting transfer payload in half without affecting vLLM bf16 serving).
    """
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "adapter"
        stage.mkdir()
        for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            src = adapter_dir / fname
            if not src.exists():
                continue
            dst = stage / fname
            if fname == "adapter_model.safetensors" and cast_bf16:
                try:
                    import torch
                    from safetensors.torch import load_file, save_file

                    tensors = load_file(str(src))
                    casted = {
                        k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype != torch.bfloat16 else v)
                        for k, v in tensors.items()
                    }
                    save_file(casted, str(dst))
                    continue
                except Exception:
                    pass
            shutil.copy2(src, dst)

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tf:
            for p in sorted(stage.iterdir()):
                tf.add(p, arcname=p.name)
        return buf.getvalue()


def push_lora_adapter(
    adapter_dir: Path,
    *,
    sync_url: str,
    lora_name: str,
    unload_prev: str | None = None,
    cast_bf16: bool = True,
    timeout: float = 180.0,
) -> dict:
    """Pack `adapter_dir`, upload to the remote LoRA sync server, and hot-reload it in vLLM."""
    _validate_name(lora_name)
    t0 = time.time()
    payload = pack_adapter_tar_gz(adapter_dir, cast_bf16=cast_bf16)
    pack_s = round(time.time() - t0, 3)

    qs = {"name": lora_name}
    if unload_prev and unload_prev != lora_name:
        _validate_name(unload_prev)
        qs["unload"] = unload_prev
    url = f"{sync_url.rstrip('/')}/sync_lora?{urllib.parse.urlencode(qs)}"

    t_up = time.time()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/gzip", "Content-Length": str(len(payload))},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    res["pack_s"] = pack_s
    res["upload_and_load_s"] = round(time.time() - t_up, 3)
    res["payload_mb"] = round(len(payload) / 1e6, 2)
    return res


def make_sync_handler(vllm_url: str, lora_root: Path):
    vllm_base = vllm_url.rstrip("/")
    lora_root = Path(lora_root)
    lora_root.mkdir(parents=True, exist_ok=True)
    state = {"active_lora": None}

    class LoRASyncHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"[lora_sync] {self.address_string()} - {fmt % args}", flush=True)

        def _send_json(self, code: int, obj: dict) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                self._send_json(200, {"ok": True, "active_lora": state["active_lora"], "vllm_url": vllm_base})
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/sync_lora":
                self._send_json(404, {"ok": False, "error": "unknown path"})
                return
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                name = _validate_name((qs.get("name") or [""])[0])
                unload_name = (qs.get("unload") or [None])[0]
                if unload_name:
                    _validate_name(unload_name)
            except ValueError as e:
                self._send_json(400, {"ok": False, "error": str(e)})
                return

            clen = int(self.headers.get("Content-Length", "0"))
            if clen <= 0:
                self._send_json(400, {"ok": False, "error": "empty body"})
                return
            raw = self.rfile.read(clen)

            target_dir = lora_root / name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        fname = Path(m.name).name
                        if fname not in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}:
                            continue
                        fobj = tf.extractfile(m)
                        if fobj is not None:
                            (target_dir / fname).write_bytes(fobj.read())
            except (tarfile.TarError, gzip.BadGzipFile, OSError) as e:
                self._send_json(400, {"ok": False, "error": f"bad tar.gz: {e}"})
                return

            if not (target_dir / "adapter_config.json").exists():
                self._send_json(400, {"ok": False, "error": "adapter_config.json missing from archive"})
                return

            # If reloading an existing name, unload it first so vLLM refreshes weights cleanly.
            _post_json(f"{vllm_base}/v1/unload_lora_adapter", {"lora_name": name}, timeout=30.0)

            t_load = time.time()
            code, resp_text = _post_json(
                f"{vllm_base}/v1/load_lora_adapter",
                {"lora_name": name, "lora_path": str(target_dir)},
                timeout=60.0,
            )
            load_ms = round((time.time() - t_load) * 1000.0, 1)
            if code != 200:
                self._send_json(502, {"ok": False, "vllm_status": code, "vllm_resp": resp_text[:500]})
                return

            if unload_name and unload_name != name:
                _post_json(f"{vllm_base}/v1/unload_lora_adapter", {"lora_name": unload_name}, timeout=30.0)

            state["active_lora"] = name
            self._send_json(
                200,
                {
                    "ok": True,
                    "lora_name": name,
                    "lora_path": str(target_dir),
                    "load_ms": load_ms,
                    "bytes": len(raw),
                },
            )

    return LoRASyncHandler


def run_sync_server(host: str, port: int, vllm_url: str, lora_root: Path) -> None:
    handler = make_sync_handler(vllm_url, lora_root)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"[lora_sync] Serving on http://{host}:{port} -> vLLM {vllm_url} (dir={lora_root})", flush=True)
    server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP LoRA hot-reload sync sidecar for vLLM")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--lora-root", default="/content/loras")
    a = ap.parse_args()
    run_sync_server(a.host, a.port, a.vllm_url, Path(a.lora_root))


if __name__ == "__main__":
    main()

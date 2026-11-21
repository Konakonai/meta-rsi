"""Local unit tests for Two-Timescale RSI training utilities (no GPU required)."""
from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from meta_rsi.llm import LLMClient
from meta_rsi.training.collect_trajectories import collect_trajectories, select_train_slice
from meta_rsi.training.lora_sync import make_sync_handler, push_lora_adapter
from meta_rsi.training.split_two_timescale import build_masked_sequence, compute_group_advantages

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "v1"


def test_group_advantages_and_zero_variance():
    # Group 1: varied rewards -> zero mean, unit-ish std; Group 2: identical rewards -> exact 0.0
    advs = compute_group_advantages([1.0, 0.0, 1.0, 0.0, 0.05, 0.05, 0.05, 0.05], group_size=4)
    assert len(advs) == 8
    assert abs(sum(advs[:4])) < 1e-6
    assert advs[0] > 0 and advs[1] < 0
    assert all(a == 0.0 for a in advs[4:])


def test_build_masked_sequence():
    inp, lab = build_masked_sequence([1, 2, 3], [10, 11, 12], max_len=10)
    assert inp == [1, 2, 3, 10, 11, 12]
    assert lab == [-100, -100, -100, 10, 11, 12]

    inp2, lab2 = build_masked_sequence([1, 2, 3], [10, 11, 12, 13], max_len=5)
    assert inp2 == [1, 2, 3, 10, 11]
    assert lab2 == [-100, -100, -100, 10, 11]


def test_lora_sync_roundtrip_with_mock_vllm():
    vllm_calls: list[tuple[str, dict]] = []

    class MockVLLM(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            clen = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(clen).decode())
            vllm_calls.append((self.path, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

    vllm_srv = ThreadingHTTPServer(("127.0.0.1", 0), MockVLLM)
    vllm_port = vllm_srv.server_address[1]
    t_vllm = threading.Thread(target=vllm_srv.serve_forever, daemon=True)
    t_vllm.start()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        lora_src = tmp_p / "src_lora"
        lora_src.mkdir()
        (lora_src / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
        (lora_src / "adapter_model.safetensors").write_bytes(b"FAKE_SAFETENSORS_BYTES")

        lora_root = tmp_p / "remote_loras"
        sync_srv = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_sync_handler(f"http://127.0.0.1:{vllm_port}", lora_root),
        )
        sync_port = sync_srv.server_address[1]
        t_sync = threading.Thread(target=sync_srv.serve_forever, daemon=True)
        t_sync.start()

        try:
            res = push_lora_adapter(
                lora_src,
                sync_url=f"http://127.0.0.1:{sync_port}",
                lora_name="grpo_s01",
                unload_prev="grpo_s00",
                cast_bf16=False,
            )
            assert res["ok"] is True
            assert res["lora_name"] == "grpo_s01"
            assert (lora_root / "grpo_s01" / "adapter_config.json").exists()
            assert (lora_root / "grpo_s01" / "adapter_model.safetensors").read_bytes() == b"FAKE_SAFETENSORS_BYTES"
            paths = [p for p, _ in vllm_calls]
            assert "/v1/load_lora_adapter" in paths
            assert any(p == "/v1/unload_lora_adapter" and b.get("lora_name") == "grpo_s00" for p, b in vllm_calls)
        finally:
            sync_srv.shutdown()
            vllm_srv.shutdown()


def test_select_train_slice_disjoint_from_dev_and_test():
    tasks = select_train_slice(DATA_ROOT, train_offset=20, train_per_family=2)
    assert len(tasks) == 36
    assert all(t["split"] == "train" for t in tasks)
    # Verify disjointness from dev slice [0:10]
    dev_tasks = select_train_slice(DATA_ROOT, train_offset=0, train_per_family=10)
    assert not ({t["task_id"] for t in tasks} & {t["task_id"] for t in dev_tasks})


def test_collect_trajectories_filters_3857_in_metamorphic_mode():
    # Mock LLM emits EPSG:3857 area script that passes equatorial `low` on some tasks but fails metamorphic check
    wm_reply = (
        "```python\n"
        "import glob, geopandas as gpd\n"
        "f = sorted(glob.glob('*.geojson') + glob.glob('*.gpkg'))[0]\n"
        "g = gpd.read_file(f).to_crs('EPSG:3857')\n"
        "print(g.geometry.area.sum())\n"
        "```"
    )
    llm = LLMClient("mock", "mock", mock_fn=lambda s, u: wm_reply)
    seed_h = Path(__file__).resolve().parents[1] / "meta_rsi" / "harnesses" / "seed.py"
    with tempfile.TemporaryDirectory() as tmp:
        out_p = Path(tmp) / "traj.jsonl"
        summ = collect_trajectories(
            harness_path=seed_h,
            data_root=DATA_ROOT,
            out_jsonl=out_p,
            llm=llm,
            filter_mode="metamorphic",
            train_offset=20,
            train_per_family=1,
            workers=2,
        )
        assert summ["n_total"] == 18
        assert summ["usage"]["calls"] == 18
        assert summ["usage"]["errors"] == 0
        assert out_p.exists()
        assert out_p.with_suffix(".summary.json").exists()

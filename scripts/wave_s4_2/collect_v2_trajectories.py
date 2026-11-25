"""Stage certified GSB v2 probes and collect disjoint SFT trajectories for Wave S4-2.

Strictly operates on `data/v2` `train[20:60]` (`15 × 40 = 600` tasks), disjoint from all
6 Wave 1A/1B/1C evolution dev splits (`[0:12]`) and the 450 unseen test tasks.

Supports two backends:
  - `--backend vertex` (default): calls Vertex AI serverless `google/gemma-4-26b-a4b-it-maas`
    with smooth pacing (`VERTEX_MIN_INTERVAL_S=3.35`, `workers=5`, `usage.errors == 0`),
    allowing pre-collection before Colab GPUs come online.
  - `--backend vllm`: calls a local or Colab vLLM server (`--task-url http://127.0.0.1:8000/v1`,
    `workers=16`).

Usage:
    nix-shell --run "python -u scripts/wave_s4_2/collect_v2_trajectories.py --backend vertex"
    python -u scripts/wave_s4_2/collect_v2_trajectories.py --backend vllm --task-url http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CERTIFIED_PROBES = [
    ("probe_r1_aligned_wm.py", REPO / "runs/v6_gsb_v2_s3b/certified_probe_r1_aligned_gem4h.py"),
    ("probe_r2_polar_laea.py", REPO / "runs/v6_gsb_v2_s3b/certified_probe_r2_gem4h.py"),
    ("probe_r3_km_units.py", REPO / "runs/v6_gsb_v2_s3b/certified_probe_r3_gem4h.py"),
]

DEFAULT_WEAK_TEACHER = REPO / "runs/v6_gsb_v2_s3/v2_sym26b_s0_darwinx_soft_best.py"
DEFAULT_META_TEACHER = REPO / "runs/v6_gsb_v2_s3b_stage2/stage2_coev_best.py"
DEFAULT_OUT_DIR = REPO / "runs/v8_s4_wave2_v2"


def stage_certified_probes(out_dir: Path) -> Path:
    probe_dir = out_dir / "certified_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for dst_name, src in CERTIFIED_PROBES:
        if not src.exists():
            raise FileNotFoundError(f"Missing certified probe: {src}")
        shutil.copy2(src, probe_dir / dst_name)
    return probe_dir


def is_clean_trajectories(jsonl_path: Path, expected_total: int) -> bool:
    summary_path = jsonl_path.with_suffix(".summary.json")
    if not jsonl_path.exists() or not summary_path.exists():
        return False
    try:
        s = json.loads(summary_path.read_text())
        u = s.get("usage") or {}
        return (
            int(s.get("n_total", 0)) >= expected_total
            and int(u.get("errors", 1)) == 0
            and int(u.get("calls", 0)) >= int(s.get("n_total", 0))
            and int(s.get("n_kept", 0)) > 0
        )
    except Exception:
        return False


def run_one(
    *,
    harness: Path,
    filter_mode: str,
    out_jsonl: Path,
    probe_dir: Path | None,
    data_root: Path,
    train_offset: int,
    train_per_family: int,
    task_url: str,
    task_model: str,
    api_key: str | None,
    workers: int,
    exec_timeout: float,
    vertex_pacing: str | None,
) -> None:
    expected_total = 15 * train_per_family
    if is_clean_trajectories(out_jsonl, expected_total):
        s = json.loads(out_jsonl.with_suffix(".summary.json").read_text())
        print(
            f"[wave2-traj] SKIP clean {out_jsonl.name}: kept={s['n_kept']}/{s['n_total']} "
            f"(3857={s['kept_3857_count']}, uerr=0)",
            flush=True,
        )
        return

    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    if vertex_pacing:
        env["VERTEX_MIN_INTERVAL_S"] = vertex_pacing

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "meta_rsi.training.collect_trajectories",
        "--harness",
        str(harness),
        "--data-root",
        str(data_root),
        "--out",
        str(out_jsonl),
        "--filter-mode",
        filter_mode,
        "--train-offset",
        str(train_offset),
        "--train-per-family",
        str(train_per_family),
        "--task-backend",
        "openai-compat",
        "--task-model",
        task_model,
        "--task-url",
        task_url,
        "--task-temp",
        "0.0",
        "--task-max-tokens",
        "2048",
        "--task-no-think",
        "--exec-timeout",
        str(exec_timeout),
        "--llm-timeout",
        "180.0",
        "--workers",
        str(workers),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    if probe_dir is not None:
        cmd.extend(["--probe-dir", str(probe_dir)])

    print(f"[wave2-traj] Collecting {out_jsonl.name} ({filter_mode}) via {task_model} ...", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect disjoint GSB v2 SFT trajectories for Wave S4-2")
    ap.add_argument("--backend", choices=["vertex", "vllm"], default="vertex")
    ap.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID", "gen-lang-client-0280760265"))
    ap.add_argument("--task-url", default="")
    ap.add_argument("--task-model", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--vertex-pacing", default=os.environ.get("VERTEX_MIN_INTERVAL_S", "3.35"))
    ap.add_argument("--data-root", default=str(REPO / "data/v2"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--weak-teacher", default=str(DEFAULT_WEAK_TEACHER))
    ap.add_argument("--meta-teacher", default=str(DEFAULT_META_TEACHER))
    ap.add_argument("--train-offset", type=int, default=20)
    ap.add_argument("--train-per-family", type=int, default=40)
    ap.add_argument("--exec-timeout", type=float, default=25.0)
    ap.add_argument("--only", choices=["both", "weak", "meta"], default="both")
    ap.add_argument("--stage-probes-only", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = stage_certified_probes(out_dir)
    print(f"[wave2-traj] Staged 3 certified probes in {probe_dir}", flush=True)
    if a.stage_probes_only:
        return

    if a.backend == "vertex":
        task_url = a.task_url or (
            f"https://aiplatform.googleapis.com/v1/projects/{a.project_id}/locations/global/endpoints/openapi"
        )
        task_model = a.task_model or "google/gemma-4-26b-a4b-it-maas"
        api_key = "gcloud"
        workers = a.workers or 5
        pacing = a.vertex_pacing
    else:
        task_url = a.task_url or "http://127.0.0.1:8000/v1"
        task_model = a.task_model or "google/gemma-4-26B-A4B-it"
        api_key = None
        workers = a.workers or 16
        pacing = None

    if a.only in {"both", "weak"}:
        run_one(
            harness=Path(a.weak_teacher),
            filter_mode="weak",
            out_jsonl=traj_dir / "weak_trajectories.jsonl",
            probe_dir=None,
            data_root=Path(a.data_root),
            train_offset=a.train_offset,
            train_per_family=a.train_per_family,
            task_url=task_url,
            task_model=task_model,
            api_key=api_key,
            workers=workers,
            exec_timeout=a.exec_timeout,
            vertex_pacing=pacing,
        )

    if a.only in {"both", "meta"}:
        run_one(
            harness=Path(a.meta_teacher),
            filter_mode="metamorphic",
            out_jsonl=traj_dir / "meta_trajectories.jsonl",
            probe_dir=probe_dir,
            data_root=Path(a.data_root),
            train_offset=a.train_offset,
            train_per_family=a.train_per_family,
            task_url=task_url,
            task_model=task_model,
            api_key=api_key,
            workers=workers,
            exec_timeout=a.exec_timeout,
            vertex_pacing=pacing,
        )

    print("[wave2-traj] ALL DONE cleanly.", flush=True)


if __name__ == "__main__":
    main()

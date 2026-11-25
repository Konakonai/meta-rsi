"""Fill missing seeds s1..s5 for GSB v2 single-seed baselines (`naive`, `meta_gate_only`, `oracle_soft`).

Produces a balanced 6-seed comparison across ALL GSB v2 evolution methods on `data/v2` (`eval450`),
matching exact Wave 1A/1B evolution hyperparameters (`dev_per_family=2`, `dev_offset=2*seed`,
`generations=6`, `soft_regress_delta=0.14`, `minimal_meta_prompt=True`, `meta_domain_hints=False`,
`task_temp=0.0`, `meta_temp=0.5`, `no_think=True`), strictly auditing `usage.errors == 0 && calls >= n`.

Supports both `--backend vertex` (Vertex AI MaaS `google/gemma-4-26b-a4b-it-maas`) and
`--backend vllm` (local / Colab vLLM server serving `google/gemma-4-26B-A4B-it`).

Usage:
    nix-shell --run "python -u scripts/rebuttal/run_v2_missing_seeds.py --backend vertex"
    python -u scripts/rebuttal/run_v2_missing_seeds.py --backend vllm --task-url http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from meta_rsi.evolve import run_evolution
from meta_rsi.llm import LLMClient
from meta_rsi.training.collect_trajectories import load_probes_from_dir

DEFAULT_OUT_DIR = REPO / "runs/v9_rebuttal_6seed"
DEFAULT_PROBE_DIR = REPO / "runs/v8_s4_wave2_v2/certified_probes"


def is_clean_eval(path: Path, expected_n: int = 450) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
        u = d.get("usage") or {}
        n = int((d.get("ALL") or {}).get("n", 0))
        return n >= expected_n and int(u.get("errors", 1)) == 0 and int(u.get("calls", 0)) >= n
    except Exception:
        return False


def run_one_seed_method(
    *,
    method: str,
    seed: int,
    out_dir: Path,
    data_root: Path,
    probe_dir: Path,
    task_url: str,
    task_model: str,
    api_key: str | None,
    workers: int,
    eval_workers: int,
    generations: int = 6,
    exec_timeout: float = 25.0,
) -> None:
    tag = f"v2_s{seed}_{method}"
    eval_json = out_dir / f"eval_{tag}.json"
    best_py = out_dir / f"{tag}_best.py"
    archive_json = out_dir / f"{tag}_archive.json"

    if is_clean_eval(eval_json, 450) and best_py.exists():
        d = json.loads(eval_json.read_text())
        print(
            f"[6seed] SKIP clean {tag}: ALL={d['ALL']['correct']*100:.1f}% "
            f"HO={d['HELDOUT_3']['correct']*100:.1f}% calls={d['avg_llm_calls']:.2f}",
            flush=True,
        )
        return

    custom_probes = None
    if method == "meta_gate_only":
        custom_probes = load_probes_from_dir(probe_dir, out_dir / "_probe_mods")
        assert custom_probes and len(custom_probes) == 3, f"Expected 3 probes in {probe_dir}"

    if not best_py.exists() or not archive_json.exists():
        t0 = time.time()
        print(f"\n[6seed] === Evolving {tag} (gate={method}, dev_offset={2*seed}, gens={generations}) ===", flush=True)
        task_llm = LLMClient(
            "openai-compat",
            task_model,
            base_url=task_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
            enable_thinking=False,
            timeout=180.0,
        )
        meta_llm = LLMClient(
            "openai-compat",
            task_model,
            base_url=task_url,
            api_key=api_key,
            temperature=0.5,
            max_tokens=8192,
            enable_thinking=False,
            timeout=180.0,
        )
        exp_dir = out_dir / "evos" / tag
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        archive = run_evolution(
            exp_dir=exp_dir,
            data_root=data_root,
            gate_mode=method,
            generations=generations,
            task_llm=task_llm,
            meta_llm=meta_llm,
            dev_per_family=2,
            dev_offset=2 * seed,
            test_seen_per_family=0,
            test_heldout_per_family=0,
            max_llm_calls=4,
            max_exec_calls=12,
            soft_regress_delta=0.14,
            parent_policy="auto",
            include_domain_hint=False,
            minimal_meta_prompt=True,
            custom_probes=custom_probes,
            exec_timeout=exec_timeout,
            workers=workers,
            seed=seed,
        )
        best = archive.best()
        shutil.copy2(best.harness_path, best_py)
        shutil.copy2(exp_dir / "archive.json", archive_json)
        print(
            f"[6seed] Evolved {tag} in {time.time()-t0:.1f}s -> best=gen_{best.gen:03d} "
            f"(in_loop={best.in_loop_rate:.1%}, ora_dev={best.oracle_dev_rate:.1%}, "
            f"uerr={task_llm.usage.errors}+{meta_llm.usage.errors})",
            flush=True,
        )
        if task_llm.usage.errors > 0 or meta_llm.usage.errors > 0:
            raise RuntimeError(f"Non-zero LLM usage errors during {tag} evolution!")

    print(f"[6seed] Evaluating {tag} on eval450 ...", flush=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "meta_rsi.eval.eval_test",
        "--harness",
        str(best_py),
        "--data-root",
        str(data_root),
        "--out",
        str(eval_json),
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
        "--max-llm-calls",
        "4",
        "--exec-timeout",
        str(exec_timeout),
        "--workers",
        str(eval_workers),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    subprocess.run(cmd, check=True, cwd=REPO)
    if not is_clean_eval(eval_json, 450):
        raise RuntimeError(f"Audit failed on {eval_json} (usage.errors != 0 or calls < n)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill missing GSB v2 seeds s1..s5 for naive/meta_gate_only/oracle_soft")
    ap.add_argument("--backend", choices=["vertex", "vllm"], default="vertex")
    ap.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID", "gen-lang-client-0280760265"))
    ap.add_argument("--task-url", default="")
    ap.add_argument("--task-model", default="")
    ap.add_argument("--methods", default="naive,meta_gate_only,oracle_soft")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-root", default=str(REPO / "data/v2"))
    ap.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--eval-workers", type=int, default=0)
    ap.add_argument("--vertex-pacing", default=os.environ.get("VERTEX_MIN_INTERVAL_S", "3.35"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.backend == "vertex":
        os.environ["VERTEX_MIN_INTERVAL_S"] = a.vertex_pacing
        task_url = a.task_url or (
            f"https://aiplatform.googleapis.com/v1/projects/{a.project_id}/locations/global/endpoints/openapi"
        )
        task_model = a.task_model or "google/gemma-4-26b-a4b-it-maas"
        api_key = "gcloud"
        workers = a.workers or 4
        eval_workers = a.eval_workers or 5
    else:
        task_url = a.task_url or "http://127.0.0.1:8000/v1"
        task_model = a.task_model or "google/gemma-4-26B-A4B-it"
        api_key = None
        workers = a.workers or 16
        eval_workers = a.eval_workers or 24

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in a.seeds.split(",") if s.strip()]

    for seed in seeds:
        for method in methods:
            run_one_seed_method(
                method=method,
                seed=seed,
                out_dir=out_dir,
                data_root=Path(a.data_root),
                probe_dir=Path(a.probe_dir),
                task_url=task_url,
                task_model=task_model,
                api_key=api_key,
                workers=workers,
                eval_workers=eval_workers,
            )

    print("[6seed] ALL REQUESTED RUNS COMPLETE & AUDITED.", flush=True)


if __name__ == "__main__":
    main()

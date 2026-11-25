"""Re-evaluate every GSB v2 6-seed best harness on one backend: Protocol A (repeated) + Protocol B.

The 6-seed table mixes eval backends (arch_coev on Vertex MaaS, every baseline on local vLLM bf16), its eval
JSONs carry no per-task records, and Protocol B (live per-variant deployment) was only run on bare weights,
never on the evolved harnesses the paper is about. This runner evaluates all method x seed harnesses against
one server with the same decoding settings and the current eval_test (which saves per-task records), so the
table can be compared seed by seed and task by task (see meta_rsi/eval/unified_summary.py).

  inventory  (local)  copy each method x seed best harness into <out>/harnesses/, write <out>/manifest.json
  run        (any)    run the manifest's eval jobs against an OpenAI-compatible server, skipping clean outputs;
                      order: Protocol A rep 1 for every harness, then Protocol B, then the extra A reps
  launch     (local)  bundle code + data/v2 + <out>, upload to a Colab session, start vLLM there, execute
                      `run`, then download <out>/evals/
  fetch      (local)  download finished eval JSONs from a session while `run` is still going

Usage:
    nix-shell --run "python scripts/rebuttal/unified_6seed_eval.py inventory"
    nix-shell --run "python -u scripts/rebuttal/unified_6seed_eval.py launch --session w3-rebuttal"
    nix-shell --run "python -m meta_rsi.eval.unified_summary runs/v10_unified_6seed"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from meta_rsi.eval.eval_test import code_sidecar  # noqa: E402

DEFAULT_OUT = REPO / "runs/v10_unified_6seed"
MODEL_ID = "google/gemma-4-26B-A4B-it"
METHODS = ["naive", "darwinx_soft", "llm_judge", "oracle_soft", "meta_gate_only", "meta_T0", "coev_tip_only", "arch_coev"]
SEEDS = range(6)

# Same search order as meta_rsi/eval/sprint4_summary.py, plus the seeds filled in by the rebuttal GPU run.
ROOTS = [
    "runs/v9_rebuttal_gpu/missing_6seed",
    "runs/v8_s4_wave2_v2",
    "runs/v7_s4_wave1c_g4b",
    "runs/v7_s4_wave1c_g4a",
    "runs/v7_s4_wave1b_g4b",
    "runs/v7_s4_wave1b_g4a",
    "runs/v7_s4_wave1a_g4b",
    "runs/v7_s4_wave1a_g4a",
    "runs/v6_gsb_v2_s3b_stage2",
    "runs/v6_gsb_v2_s3b",
    "runs/v6_gsb_v2_s3",
]
# s0 runs predate the eval_v2_s{seed}_{method}.json / v2_s{seed}_{method}_best.py naming.
S0_SPECIAL = {
    "darwinx_soft": ("eval450_v2_sym26b_s0_darwinx_soft.json", "v2_sym26b_s0_darwinx_soft_best.py"),
    "meta_T0": ("eval450_stage1_meta_T0.json", "stage1_best.py"),
    "coev_tip_only": ("eval450_stage2_coev.json", "stage2_coev_best.py"),
}


def _eval_names(method: str, seed: int) -> list[str]:
    if method == "coev_tip_only":
        return [f"eval_v2_s{seed}_coev_v2.json", f"eval_v2_s{seed}_coev.json"]
    return [f"eval_v2_s{seed}_{method}.json"]


def find_best(method: str, seed: int) -> tuple[Path, Path] | None:
    """(original eval JSON, best harness) for one method x seed, picking the eval the 6-seed table uses."""
    pairs: list[tuple[str, str]] = []
    if seed == 0 and method in S0_SPECIAL:
        pairs.append(S0_SPECIAL[method])
    pairs += [(e, e.removeprefix("eval_").removesuffix(".json") + "_best.py") for e in _eval_names(method, seed)]
    for ev, hv in pairs:  # candidate names outer, roots inner, as in sprint4_summary._find_first
        for root in ROOTS:
            if (REPO / root / ev).exists():
                harness = REPO / root / hv
                return (REPO / root / ev, harness) if harness.exists() else None
    return None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def is_clean(path: Path, min_n: int) -> bool:
    """Complete, error-free, and carries per-task records."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    u = d.get("usage") or {}
    n = int((d.get("ALL") or {}).get("n", 0))
    ok = n >= min_n and int(u.get("errors", 1)) == 0 and int(u.get("calls", 0)) >= n and len(d.get("records") or []) == n
    if ok and d.get("protocol", "A") == "A":  # Protocol A also needs its scripts, for blind-variant scoring
        cp = code_sidecar(path)
        ok = cp.exists() and sum(1 for _ in open(cp)) == n
    return ok


def cmd_inventory(a: argparse.Namespace) -> None:
    from meta_rsi.sandbox import validate_harness_source

    out = Path(a.out)
    (out / "harnesses").mkdir(parents=True, exist_ok=True)
    (out / "evals").mkdir(parents=True, exist_ok=True)
    methods = a.methods.split(",") if a.methods else METHODS
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS)
    entries, missing = [], []
    for m in methods:
        for s in seeds:
            tag = f"{m}_s{s}"
            found = find_best(m, s)
            if found is None:
                missing.append(tag)
                continue
            ev, harness = found
            ok, why = validate_harness_source(harness.read_text())
            if not ok:
                raise SystemExit(f"{tag}: {harness} fails the harness check: {why}")
            dst = out / "harnesses" / f"{tag}.py"
            shutil.copy2(harness, dst)
            orig = json.loads(ev.read_text())
            entries.append({
                "tag": tag, "method": m, "seed": s,
                "harness": str(dst.relative_to(out)), "sha256": _sha(dst),
                "source": str(harness.relative_to(REPO)),
                "orig_eval": str(ev.relative_to(REPO)),
                "orig_model": orig.get("model"),
                "orig_correct": (orig.get("ALL") or {}).get("correct"),
            })
            # An existing eval made with records on the same server settings counts as Protocol A rep 1.
            reuse = out / "evals" / f"{tag}_A_rep1.json"
            if (orig.get("records") and orig.get("model") == a.model and orig.get("temp") == 0.0
                    and orig.get("no_think") and orig.get("protocol", "A") == "A" and not reuse.exists()
                    and code_sidecar(ev).exists()):
                shutil.copy2(ev, reuse)
                shutil.copy2(code_sidecar(ev), code_sidecar(reuse))
                entries[-1]["reused_A_rep1"] = True
    manifest = {"model": a.model, "data_root": "data/v2", "entries": entries, "missing": missing}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    by_m = {m: sum(e["method"] == m for e in entries) for m in methods}
    print(f"[unified] {len(entries)} harnesses -> {out / 'manifest.json'} {by_m}")
    if missing:
        print(f"[unified] MISSING {len(missing)}: {', '.join(missing)}")


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _start_vllm(model: str, log_path: Path) -> subprocess.Popen:
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints|VLLM::|EngineCore"], check=False)
    time.sleep(3)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--host", "127.0.0.1", "--port", "8000",
        "--gpu-memory-utilization", "0.86", "--max-model-len", "8192", "--max-num-seqs", "64",
    ]
    print(f"[unified] launching vLLM ({model}) -> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while not _http_ok("http://127.0.0.1:8000/v1/models"):
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with rc={proc.returncode}; see {log_path}")
        if time.time() - t0 > 900:
            raise TimeoutError("Timed out waiting for vLLM")
        time.sleep(5)
    print(f"[unified] vLLM ready in {time.time() - t0:.0f}s", flush=True)
    return proc


def plan_jobs(entries: list[dict], a_reps: int, b_reps: int) -> list[tuple[dict, str, int]]:
    """A rep 1 for everything first (the table), then B (the live check), then the variance reps."""
    jobs = [(e, "A", 1) for e in entries]
    jobs += [(e, "B", r) for r in range(1, b_reps + 1) for e in entries]
    jobs += [(e, "A", r) for r in range(2, a_reps + 1) for e in entries]
    return jobs


def cmd_run(a: argparse.Namespace) -> None:
    out = Path(a.out).resolve()  # eval_test runs with cwd=REPO
    manifest = json.loads((out / "manifest.json").read_text())
    data_root = REPO / manifest["data_root"]
    rows = [json.loads(ln) for ln in open(data_root / "tasks.jsonl")]
    per: dict[str, int] = {}
    for r in rows:
        if r["split"] == "test":
            per[r["family"]] = per.get(r["family"], 0) + 1

    def cap(proto: str) -> int:  # tasks per family (0 = all) for a protocol
        return a.b_per_family if proto == "B" and a.b_per_family else a.per_family

    def expected_n(proto: str) -> int:
        return sum(min(c, cap(proto)) if cap(proto) else c for c in per.values())

    jobs = plan_jobs(manifest["entries"], a.a_reps, a.b_reps)

    vllm = _start_vllm(manifest["model"], out / "vllm_unified.log") if a.start_vllm else None
    state = {"i": 0, "t0": time.time()}
    stop = threading.Event()

    def _hb() -> None:  # `colab exec --timeout` is an idle limit; one eval prints nothing for minutes
        while not stop.wait(60.0):
            print(f"[unified hb] {time.time() - state['t0']:.0f}s job {state['i']}/{len(jobs)}", flush=True)

    threading.Thread(target=_hb, daemon=True).start()
    failed = []
    try:
        for i, (e, proto, rep) in enumerate(jobs, 1):
            state["i"] = i
            dst = out / "evals" / f"{e['tag']}_{proto}_rep{rep}.json"
            if is_clean(dst, expected_n(proto)):
                continue
            cmd = [
                sys.executable, "-u", "-m", "meta_rsi.eval.eval_test",
                "--harness", str(out / e["harness"]), "--data-root", str(data_root),
                "--protocol", proto, "--out", str(dst),
                "--task-backend", "openai-compat", "--task-model", manifest["model"], "--task-url", a.task_url,
                "--task-temp", "0.0", "--task-max-tokens", "2048", "--task-no-think",
                "--max-llm-calls", "4", "--exec-timeout", "25.0", "--workers", str(a.workers),
            ]
            if cap(proto):  # eval_test takes the first k test tasks of each family: same subset for every harness
                cmd += ["--seen-per-family", str(cap(proto)), "--heldout-per-family", str(cap(proto))]
            for attempt in (1, 2):
                print(f"[unified] job {i}/{len(jobs)} {dst.name} (attempt {attempt})", flush=True)
                rc = subprocess.run(cmd, cwd=REPO).returncode
                if rc == 0 and is_clean(dst, expected_n(proto)):
                    break
            else:
                failed.append(dst.name)
                print(f"[unified] FAILED {dst.name} (rc={rc}); a rerun retries it", flush=True)
    finally:
        stop.set()
        if vllm is not None:
            subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints|VLLM::|EngineCore"], check=False)
    done = sum(is_clean(out / "evals" / f"{e['tag']}_{p}_rep{r}.json", expected_n(p)) for e, p, r in jobs)
    print(f"[unified] {done}/{len(jobs)} jobs clean in {time.time() - state['t0']:.0f}s; failed: {failed or 'none'}", flush=True)
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------- Colab
REMOTE_CELL = r'''
import os, subprocess, sys, tarfile
from pathlib import Path
parts = sorted(Path("/content").glob("unified_bundle.part*"))
with open("/content/unified_bundle.tgz", "wb") as f:
    for p in parts:
        f.write(p.read_bytes()); p.unlink()
work = Path("/content/unified_work"); work.mkdir(exist_ok=True)
with tarfile.open("/content/unified_bundle.tgz") as tf:
    tf.extractall(work)  # overwrites code and harnesses; remote evals not in the bundle are kept
os.chdir(work)
need = []
for mod in ("geopandas", "pyogrio", "pyproj", "shapely", "rasterio", "scipy", "vllm"):
    try:
        __import__(mod)
    except Exception:
        need.append(mod)
if need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
# Relay the runner's output: a child's own stdout never reaches the notebook stream, and `colab exec`
# fails after --timeout seconds without output.
proc = subprocess.Popen([sys.executable, "-u", "scripts/rebuttal/unified_6seed_eval.py", "--out", "__OUT__", "run",
                         "--start-vllm", "--a-reps", "__A_REPS__", "--b-reps", "__B_REPS__", "--workers", "__WORKERS__",
                         "--b-per-family", "__B_PF__"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
with tarfile.open("/content/unified_out.tgz", "w:gz") as tf:
    tf.add("__OUT__/evals", arcname="evals")
print(f"[unified] remote run rc={rc}; packed /content/unified_out.tgz", flush=True)
'''


def _colab(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    from scripts.wave_s4_2.run_single_g4_wave2 import _run_colab
    return _run_colab(args, check=check)


def cmd_launch(a: argparse.Namespace) -> None:
    from scripts.wave_s4_2.run_single_g4_wave2 import upload_chunked

    out = Path(a.out).resolve()
    rel_out = out.relative_to(REPO)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["missing"] and not a.allow_missing:
        raise SystemExit(f"manifest is missing {manifest['missing']}; rerun inventory or pass --allow-missing")
    data_root = REPO / manifest["data_root"]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "unified_bundle.tgz"
        with tarfile.open(bundle, "w:gz") as tf:
            for top in ("meta_rsi", "benchmarks"):
                for p in (REPO / top).rglob("*.py"):
                    if "__pycache__" not in p.parts:
                        tf.add(p, arcname=str(p.relative_to(REPO)))
            tf.add(Path(__file__), arcname="scripts/rebuttal/unified_6seed_eval.py")
            # Full tasks.jsonl (train rows decide SEEN vs HELDOUT) but only the test workspaces.
            tf.add(data_root / "tasks.jsonl", arcname=f"{manifest['data_root']}/tasks.jsonl")
            dirs = {v["dir"] for ln in open(data_root / "tasks.jsonl") for t in [json.loads(ln)]
                    if t["split"] == "test" for v in t["variants"].values()}
            for d in sorted(dirs):
                tf.add(data_root / d, arcname=f"{manifest['data_root']}/{d}")
            tf.add(out / "manifest.json", arcname=f"{rel_out}/manifest.json")
            tf.add(out / "harnesses", arcname=f"{rel_out}/harnesses")
            tf.add(out / "evals", arcname=f"{rel_out}/evals")
        print(f"[unified] bundle {bundle.stat().st_size / 1e6:.1f} MB", flush=True)
        upload_chunked(a.session, bundle, "/content/unified_bundle.part")
        cell = Path(tmp) / "unified_remote.py"
        cell.write_text(
            REMOTE_CELL.replace("__OUT__", str(rel_out)).replace("__A_REPS__", str(a.a_reps))
            .replace("__B_REPS__", str(a.b_reps)).replace("__WORKERS__", str(a.workers)).replace("__B_PF__", str(a.b_per_family))
        )
        _colab(["exec", "-s", a.session, "--timeout", "1800", "-f", str(cell)], check=False)
        local_tgz = Path(tmp) / "unified_out.tgz"
        _colab(["download", "-s", a.session, "/content/unified_out.tgz", str(local_tgz)])
        with tarfile.open(local_tgz) as tf:
            tf.extractall(out, filter="data")
    print(f"[unified] results in {out / 'evals'}", flush=True)


def cmd_fetch(a: argparse.Namespace) -> None:
    out = Path(a.out).resolve()
    remote = f"/content/unified_work/{out.relative_to(REPO)}/evals"
    listing = subprocess.run(["colab", "ls", "-s", a.session, remote], capture_output=True, text=True, check=True).stdout
    names = [ln.strip() for ln in listing.splitlines()
             if ln.strip().endswith((".json", ".code.jsonl"))
             and not ln.strip().endswith((".blind.json", ".heldout.json"))]
    got = 0
    for name in sorted(names, key=lambda n: n.endswith(".json")):  # sidecars first, so is_clean can see them
        dst = out / "evals" / name
        if dst.exists() and (name.endswith(".code.jsonl") or is_clean(dst, 1)):
            continue
        _colab(["download", "-s", a.session, f"{remote}/{name}", str(dst)], check=False)
        got += 1
    print(f"[unified] fetched {got} of {len(names)} remote eval files", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--methods", default="")
    p.add_argument("--seeds", default="")
    p.add_argument("--model", default=MODEL_ID)
    p = sub.add_parser("run")
    p.add_argument("--task-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--start-vllm", action="store_true", help="(re)start a local vLLM server for the manifest model")
    p.add_argument("--a-reps", type=int, default=1)
    p.add_argument("--b-reps", type=int, default=1)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--per-family", type=int, default=0, help="smoke runs: cap test tasks per family")
    p.add_argument("--b-per-family", type=int, default=0, help="Protocol B on the first k test tasks per family (0 = all)")
    p = sub.add_parser("launch")
    p.add_argument("--session", required=True)
    p.add_argument("--a-reps", type=int, default=1)
    p.add_argument("--b-reps", type=int, default=1)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--b-per-family", type=int, default=0)
    p.add_argument("--allow-missing", action="store_true")
    p = sub.add_parser("fetch")
    p.add_argument("--session", required=True)
    a = ap.parse_args()
    {"inventory": cmd_inventory, "run": cmd_run, "launch": cmd_launch, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()

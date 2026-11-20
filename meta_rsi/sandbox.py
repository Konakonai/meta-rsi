"""Safe loading and sandboxed execution helpers for self-evolving Agent Harnesses."""
from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

FORBIDDEN_IMPORTS = {
    "socket",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
    "pty",
    "webbrowser",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}


def validate_harness_source(source: str) -> tuple[bool, str]:
    """Static AST check: valid Python, defines solve(task_prompt, work_dir, llm_call, run_python), no banned ops."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    solve_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            solve_fn = node
            break
    if solve_fn is None:
        return False, "Missing top-level function `solve(task_prompt, work_dir, llm_call, run_python)`."
    if len(solve_fn.args.args) < 4:
        return False, f"`solve` must accept 4 positional args, got {len(solve_fn.args.args)}."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                return False, f"Forbidden import-from: {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return False, f"Forbidden builtin call: {node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr in ("system", "popen", "fork", "kill"):
            return False, f"Forbidden attribute access: .{node.attr}"
    return True, "ok"


def load_harness_module(path: Path):
    source = path.read_text()
    ok, reason = validate_harness_source(source)
    if not ok:
        raise ValueError(f"Invalid harness {path}: {reason}")
    mod_name = f"_rsi_harness_{path.stem}_{abs(hash((str(path), source))) % 10**9}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "solve", None)):
        raise ValueError(f"Harness {path} has non-callable `solve`")
    return mod


_SWALLOW_RUNNER = r'''
import atexit, json, os, runpy, sys

caught = []
mon = getattr(sys, "monitoring", None)
if mon is not None:
    TOOL = 5
    try:
        mon.use_tool_id(TOOL, "meta_rsi_swallow")
    except ValueError:
        pass

    def _on_handled(code, offset, exc):
        if code.co_filename.endswith("solution.py") and len(caught) < 20:
            caught.append(f"{type(exc).__name__}: {str(exc)[:120]}")

    mon.register_callback(TOOL, mon.events.EXCEPTION_HANDLED, _on_handled)
    mon.set_events(TOOL, mon.events.EXCEPTION_HANDLED)


def _dump():
    out_p = os.environ.get("META_RSI_CAUGHT_OUT")
    if out_p:
        with open(out_p, "w") as f:
            json.dump(caught, f)


atexit.register(_dump)
sys.argv = ["solution.py"]
runpy.run_path("solution.py", run_name="__main__")
'''


def run_python_sandbox(
    code: str,
    work_dir: Path,
    timeout: float = 25.0,
    python: str = sys.executable,
    *,
    track_caught: bool = False,
) -> dict:
    """Run candidate Python script in a fresh temporary copy of `work_dir`.

    When `track_caught=True` (Python 3.12+), instruments `solution.py` with
    `sys.monitoring` (`EXCEPTION_HANDLED`) to capture swallowed runtime exceptions.
    """
    import json
    import os

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        copy_dir = Path(tmp) / "work"
        shutil.copytree(work_dir, copy_dir)
        (copy_dir / "solution.py").write_text(code)
        cmd = [python, "solution.py"]
        env = None
        caught_out = Path(tmp) / "caught.json"
        if track_caught:
            (copy_dir / "_meta_rsi_runner.py").write_text(_SWALLOW_RUNNER)
            cmd = [python, "_meta_rsi_runner.py"]
            env = dict(os.environ, META_RSI_CAUGHT_OUT=str(caught_out))
        try:
            r = subprocess.run(
                cmd,
                cwd=copy_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            elapsed = round(time.time() - t0, 3)
            caught: list[str] = []
            if track_caught and caught_out.exists():
                try:
                    caught = json.loads(caught_out.read_text())
                except Exception:
                    caught = []
            res = {
                "returncode": r.returncode,
                "stdout": r.stdout[-4000:],
                "stderr": r.stderr[-4000:],
                "timed_out": False,
                "elapsed_s": elapsed,
            }
            if track_caught:
                res["caught_exceptions"] = caught
            return res
        except subprocess.TimeoutExpired as e:
            elapsed = round(time.time() - t0, 3)
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            res = {
                "returncode": -1,
                "stdout": out[-2000:],
                "stderr": (err[-2000:] + f"\nTimeoutExpired({timeout}s)").strip(),
                "timed_out": True,
                "elapsed_s": elapsed,
            }
            if track_caught:
                res["caught_exceptions"] = []
            return res


def parse_final_number(stdout: str) -> float | None:
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        val = float(lines[-1].replace(",", ""))
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return val
    except ValueError:
        return None


def run_harness_on_workspace(
    harness_mod,
    task_prompt: str,
    src_work_dir: Path,
    llm_complete: Callable[[str, str], str],
    *,
    max_llm_calls: int = 4,
    max_exec_calls: int = 12,
    exec_timeout: float = 25.0,
    python: str = sys.executable,
) -> dict:
    """Invoke `harness_mod.solve(...)` with strict LLM & execution call-budget and workspace isolation."""
    call_counter = {"n": 0, "exec": 0}

    def guarded_llm(system: str, user: str) -> str:
        if call_counter["n"] >= max_llm_calls:
            raise RuntimeError(f"Harness exceeded max_llm_calls={max_llm_calls}")
        call_counter["n"] += 1
        return llm_complete(system, user)

    def guarded_run(code: str, wdir: Path | str | None = None, timeout: float = exec_timeout) -> dict:
        if call_counter["exec"] >= max_exec_calls:
            raise RuntimeError(f"Harness exceeded max_exec_calls={max_exec_calls}")
        call_counter["exec"] += 1
        target = Path(wdir) if wdir is not None else work_copy
        eff_timeout = min(float(timeout), exec_timeout)
        return run_python_sandbox(code, target, timeout=eff_timeout, python=python)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        work_copy = Path(tmp) / "work"
        shutil.copytree(src_work_dir, work_copy)
        try:
            out = harness_mod.solve(task_prompt, work_copy, guarded_llm, guarded_run)
        except Exception as e:
            return {
                "code": None,
                "harness_error": f"{type(e).__name__}: {e}",
                "llm_calls": call_counter["n"],
                "elapsed_s": round(time.time() - t0, 3),
                "trace": [],
            }

    if not isinstance(out, dict):
        return {
            "code": None,
            "harness_error": f"solve() returned {type(out).__name__}, expected dict",
            "llm_calls": call_counter["n"],
            "elapsed_s": round(time.time() - t0, 3),
            "trace": [],
        }
    return {
        "code": out.get("code"),
        "harness_error": None,
        "llm_calls": call_counter["n"],
        "elapsed_s": round(time.time() - t0, 3),
        "trace": out.get("trace") or [],
    }

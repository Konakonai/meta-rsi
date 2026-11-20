"""Generation-0 seed Agent Harness (editable organism for RSI evolution).

Contract (MUST be preserved across all mutated versions):
    def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict

Arguments provided by the runner:
    task_prompt : str
        Natural-language problem description.
    work_dir : Path
        Isolated copy of the task's input workspace (read-only recommended for inspection;
        execution inside `run_python` always runs on its own clean sandbox copy).
    llm_call(system: str, user: str) -> str
        Calls the frozen task LLM (budget-capped by the outer runner, default <= 4 calls).
    run_python(code: str, work_dir: Path, timeout: float = 25.0) -> dict
        Executes `code` in a sandboxed copy of `work_dir` and returns:
        {"returncode": int, "stdout": str, "stderr": str, "timed_out": bool}

Return value:
    {"code": str | None, "trace": list[dict], "turns": int}
"""
from __future__ import annotations

import re
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a Python data analysis assistant. "
    "Write a single self-contained Python script inside one ```python ... ``` block "
    "that reads the files in the current working directory and prints only the final number."
)


def extract_code(text: str) -> str | None:
    final = text or ""
    for end in ("</think>", "<channel|>"):
        if end in final:
            final = final.rsplit(end, 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", final, re.S)
    if not blocks:
        return None
    return blocks[-1].strip()


def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict:
    """Gen-0 baseline: naive single-shot code generation with no inspection or repair."""
    reply = llm_call(SYSTEM_PROMPT, task_prompt)
    code = extract_code(reply)
    return {
        "code": code,
        "trace": [{"turn": 0, "has_code": code is not None}],
        "turns": 1,
    }

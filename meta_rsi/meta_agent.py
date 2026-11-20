"""Meta-Agent proposer: inspects failure & metamorphic diagnostics and rewrites harness.py."""
from __future__ import annotations

import re
from pathlib import Path

from .archive import ArchiveEntry
from .llm import LLMClient
from .sandbox import validate_harness_source

META_SYSTEM_PROMPT = """You are an AI systems researcher evolving a self-contained Python Agent Harness (`harness.py`)
that orchestrates a frozen LLM to solve verifiable Python data-analysis coding tasks reliably.

Your job is to propose ONE focused, high-leverage structural improvement to `harness.py` based
STRICTLY on the observed in-loop failures and diagnostics below (never hardcode task IDs, filenames,
or specific answers).

Hard contract for `harness.py`:
1. Must be valid, self-contained Python using ONLY stdlib plus optionally `geopandas`, `pyproj`, `shapely`, `pandas`, `numpy`.
2. Must expose: `def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict`
   returning `{"code": str | None, "trace": list[dict], "turns": int}`.
3. `llm_call(system: str, user: str) -> str` calls the frozen Task LLM (capped at <= 4 calls per task).
4. `run_python(code: str, work_dir: Path, timeout: float = 25.0) -> dict` executes code in a clean
   sandbox copy of `work_dir` and returns `{"returncode", "stdout", "stderr", "timed_out", "elapsed_s"}`.
5. Do NOT import networking modules (`socket`, `urllib`, `http`), `ctypes`, `multiprocessing`,
   and do NOT use `eval`, `exec`, `compile`, `__import__`, `os.system`, or `os.popen`.
6. Keep `harness.py` clean and concise (under ~200 lines) and keep Task LLM prompts short & modular:
   dumping long rigid rules into a small model causes skill contamination and runtime crashes on unrelated tasks.
   Prefer general agent mechanisms: workspace inspection, execute-inspect-repair loops, targeted conditional guidance,
   and lightweight verification/sanity checks.

Output format (strict):
HYPOTHESIS: <2-3 concise sentences explaining the root failure pattern and what structural change fixes it without regressing solved tasks>

```python
<complete updated harness.py>
```
"""


MINIMAL_META_SYSTEM_PROMPT = """You are an AI systems researcher evolving a self-contained Python Agent Harness (`harness.py`)
that orchestrates a frozen LLM to solve verifiable Python data-analysis coding tasks reliably.

Your job is to propose ONE focused, high-leverage improvement to `harness.py` based
STRICTLY on the observed in-loop failures and diagnostics below (never hardcode task IDs, filenames,
or specific answers).

Hard contract for `harness.py`:
1. Must be valid, self-contained Python using stdlib plus installed Python data/scientific packages.
2. Must expose: `def solve(task_prompt: str, work_dir: Path, llm_call, run_python) -> dict`
   returning `{"code": str | None, "trace": list[dict], "turns": int}`.
3. `llm_call(system: str, user: str) -> str` calls the frozen Task LLM (capped at <= 4 calls per task).
4. `run_python(code: str, work_dir: Path, timeout: float = 25.0) -> dict` executes code in a clean
   sandbox copy of `work_dir` and returns `{"returncode", "stdout", "stderr", "timed_out", "elapsed_s"}`.
5. Do NOT import networking modules (`socket`, `urllib`, `http`), `ctypes`, `multiprocessing`,
   and do NOT use `eval`, `exec`, `compile`, `__import__`, `os.system`, or `os.popen`.
6. Keep `harness.py` clean and self-contained (under ~250 lines).

Output format (strict):
HYPOTHESIS: <2-3 concise sentences explaining the root failure pattern and what change fixes it without regressing solved tasks>

```python
<complete updated harness.py>
```
"""


def _stratified_pick(records: list[dict], k: int, seed: int = 0) -> list[dict]:
    """Select up to `k` failure records spread evenly across distinct families (avoids alphabetical starvation)."""
    if len(records) <= k:
        return list(records)
    by_fam: dict[str, list[dict]] = {}
    for r in records:
        by_fam.setdefault(r.get("family", "unknown"), []).append(r)
    fams = sorted(by_fam.keys())
    if fams:
        shift = seed % len(fams)
        fams = fams[shift:] + fams[:shift]
    out: list[dict] = []
    idx = 0
    while len(out) < k:
        added = False
        for fam in fams:
            bucket = by_fam[fam]
            if idx < len(bucket):
                out.append(bucket[idx])
                added = True
                if len(out) >= k:
                    break
        if not added:
            break
        idx += 1
    return out


def build_failure_digest(
    eval_result: dict,
    gate_mode: str,
    *,
    max_examples: int = 6,
    minimal_prompt: bool = False,
    hide_meta_diag: bool = False,
    pick_seed: int = 0,
) -> str:
    """Build a strictly label-free / dev-only diagnostic digest (ZERO oracle leakage)."""
    s = eval_result["summary"]
    records = eval_result["records"]
    lines = [
        f"In-loop summary ({s['n_tasks']} dev tasks):",
        f"- dev_low pass rate: {s['dev_low_rate']:.1%}",
        f"- dev runtime/no-code error rate: {s['dev_error_rate']:.1%}",
        f"- avg LLM calls / task: {s['avg_llm_calls']:.2f}",
    ]
    if gate_mode in ("metamorphic", "meta_soft") and not hide_meta_diag and s["meta_pass_rate"] is not None:
        lines.append(f"- invariance-verified pass rate (dev_low AND invariance-consistent): {s['meta_pass_rate']:.1%}")
    elif gate_mode == "meta_gate_only" and s["meta_pass_rate"] is not None:
        lines.append(f"- verified in-loop pass rate: {s['in_loop_rate']:.1%}")
    elif gate_mode == "oracle":
        lines.append(f"- oracle 3-location pass rate (upper-bound verifier): {s['in_loop_rate']:.1%}")
    if s.get("dev_swallowed_rate") is not None:
        lines.append(f"- swallowed-exception rate among runnable scripts: {s['dev_swallowed_rate']:.1%}")

    # Prioritize: (1) invariance/multi-location failures (unless hidden for pure gate-only ablation),
    # (2) runtime crashes, (3) wrong numeric outputs — stratified across ALL families!
    if gate_mode == "oracle":
        meta_fails = [r for r in records if r["dev_low_ok"] and not r["oracle_correct"]]
    elif hide_meta_diag or gate_mode == "meta_gate_only":
        meta_fails = []
    else:
        meta_fails = [r for r in records if r["dev_low_ok"] and r["meta_checked"] and not r["meta_consistent"]]
    err_fails = [r for r in records if r["dev_low_status"] != "ok"]
    wrong_fails = [r for r in records if r["dev_low_status"] == "ok" and not r["dev_low_ok"]]

    if hide_meta_diag or gate_mode == "meta_gate_only":
        lines.append(
            f"Breakdown of in-loop failures: {len(err_fails)} runtime/format errors, "
            f"{len(wrong_fails)} wrong numeric answers on dev."
        )
        picked = _stratified_pick(err_fails, 3, pick_seed) + _stratified_pick(wrong_fails, 3, pick_seed + 1)
    else:
        fail_label = "oracle multi-location failures (lucky hacks)" if gate_mode == "oracle" else "invariance-inconsistency (silent drift)"
        lines.append(
            f"Breakdown of in-loop failures: {len(meta_fails)} {fail_label}, "
            f"{len(err_fails)} runtime/format errors, {len(wrong_fails)} wrong numeric answers on dev."
        )
        picked = (
            _stratified_pick(meta_fails, 3, pick_seed)
            + _stratified_pick(err_fails, 2, pick_seed + 1)
            + _stratified_pick(wrong_fails, 2, pick_seed + 2)
        )[:max_examples]
        if len(picked) < max_examples:
            seen_ids = {r["task_id"] for r in picked}
            rem = [r for r in (meta_fails + err_fails + wrong_fails) if r["task_id"] not in seen_ids]
            picked.extend(_stratified_pick(rem, max_examples - len(picked), pick_seed + 3))

    for i, r in enumerate(picked[:max_examples], 1):
        if minimal_prompt:
            lines.append(f"\n--- Failure Example {i} (status={r['dev_low_status']}) ---")
        else:
            lines.append(f"\n--- Failure Example {i} (family={r['family']}, status={r['dev_low_status']}) ---")
        if gate_mode == "oracle" and r["dev_low_ok"] and not r["oracle_correct"]:
            lines.append(
                f"Diagnostic: Passed single-location dev_low, but failed multi-location oracle "
                f"(category={r.get('oracle_category')})."
            )
        elif (not hide_meta_diag and gate_mode != "meta_gate_only") and r["meta_checked"] and not r["meta_consistent"]:
            lines.append(f"Diagnostic: {r['meta_diag']}")
        elif r["dev_low_status"] != "ok":
            lines.append(f"Error: {r['dev_low_stderr'][:300]}")
        else:
            lines.append(
                f"Numeric mismatch on dev: got={r['dev_low_value']}, expected={r['dev_low_truth']:.6g}"
            )
        if r.get("dev_caught_exceptions"):
            lines.append(f"Swallowed exceptions at runtime: {r['dev_caught_exceptions'][:3]}")
        snippet = (r.get("code") or "")[:900]
        if snippet:
            lines.append(f"Emitted code snippet:\n```python\n{snippet}\n```")

    return "\n".join(lines)


def _extract_hypothesis_and_code(text: str) -> tuple[str, str | None]:
    # Strip ONLY a leading reasoning preamble before `HYPOTHESIS:` / first code fence.
    # CRITICAL: Never `rsplit("<channel|>")` across the entire response, because `harness.py`
    # itself contains `for end in ("</think>", "<channel|>"):` inside its own `extract_code()`!
    clean = (text or "").lstrip()
    first_fence = clean.find("```")
    prefix = clean[:first_fence] if first_fence != -1 else clean
    for end in ("</think>", "<channel|>"):
        if end in prefix:
            cut = prefix.rfind(end) + len(end)
            clean = clean[cut:].lstrip()
            break
    # If a `thought\n...` preamble precedes `HYPOTHESIS:` before the first code fence, skip to `HYPOTHESIS:`.
    first_fence = clean.find("```")
    prefix = clean[:first_fence] if first_fence != -1 else clean
    if "HYPOTHESIS:" in prefix:
        clean = clean[prefix.rfind("HYPOTHESIS:") :]

    hyp = ""
    m_hyp = re.search(r"HYPOTHESIS:\s*(.+?)(?=\n```|\n<<<HARNESS_CODE>>>|\Z)", clean, re.S)
    if m_hyp:
        hyp = m_hyp.group(1).strip()

    # Prefer explicit sentinel markers if present.
    m_tag = re.search(r"<<<HARNESS_CODE>>>\s*\n(.*?)\n<<<END_HARNESS_CODE>>>", clean, re.S)
    if m_tag:
        return hyp, m_tag.group(1).strip()

    # Robust line-anchored fenced block extraction: handles triple backticks inside Python string literals.
    lines = clean.splitlines()
    start_idx = None
    end_idx = None
    for i, ln in enumerate(lines):
        if start_idx is None and re.match(r"^\s*```(?:python|py)?\s*$", ln):
            start_idx = i + 1
        elif start_idx is not None and re.match(r"^\s*```\s*$", ln):
            end_idx = i
    if start_idx is not None and end_idx is not None and end_idx > start_idx:
        return hyp, "\n".join(lines[start_idx:end_idx]).strip()
    return hyp, None


def propose_mutation(
    meta_llm: LLMClient,
    parent: ArchiveEntry,
    parent_eval: dict,
    recent_attempts: list[dict],
    gate_mode: str,
    *,
    partner: ArchiveEntry | None = None,
    max_retries: int = 2,
    minimal_prompt: bool = False,
    hide_meta_diag: bool = False,
    pick_seed: int = 0,
) -> tuple[str, str]:
    """Ask Meta-Agent to propose a validated new `harness.py` source + hypothesis."""
    parent_src = Path(parent.harness_path).read_text()
    digest = build_failure_digest(
        parent_eval,
        gate_mode,
        minimal_prompt=minimal_prompt,
        hide_meta_diag=hide_meta_diag,
        pick_seed=pick_seed,
    )

    hist_lines = []
    for a in recent_attempts[-6:]:
        status = "ACCEPTED" if a["accepted"] else "REJECTED"
        hist_lines.append(
            f"- Gen {a['gen']} (from {a['parent_gen']}): {status} | {a['gate_reason']} | Hypothesis: {a['rationale'][:160]}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "(none yet — this is the first mutation)"

    parts = [
        f"## Current Parent Harness (Gen {parent.gen}, in_loop={parent.in_loop_rate:.1%}, dev_err={parent.dev_error_rate:.1%})",
        f"```python\n{parent_src}\n```",
        "\n## In-Loop Diagnostic Report",
        digest,
        "\n## Recent Evolution History (avoid repeating rejected ideas!)",
        history_block,
    ]
    if partner is not None:
        partner_src = Path(partner.harness_path).read_text()
        comp_count = len(set(partner.solved_ids) - set(parent.solved_ids))
        parts.extend(
            [
                f"\n## Optional Recombination Partner (Gen {partner.gen}, solves {comp_count} tasks parent missed)",
                f"```python\n{partner_src[:12000]}\n```",
                "You may selectively combine complementary strengths from this partner if helpful.",
            ]
        )
    parts.append(
        "\nNow propose ONE targeted, clean evolution of `harness.py` that fixes the dominant failure mode "
        "without regressing already-solved tasks. Remember to preserve the exact `solve(...)` signature."
    )
    user_prompt = "\n".join(parts)
    sys_prompt = MINIMAL_META_SYSTEM_PROMPT if minimal_prompt else META_SYSTEM_PROMPT

    last_err = "no code returned"
    for attempt in range(max_retries + 1):
        prompt = user_prompt
        if attempt > 0:
            prompt += f"\n\nPREVIOUS PROPOSAL FAILED VALIDATION: {last_err}\nPlease fix and return a complete valid `harness.py`."
        resp = meta_llm.complete(sys_prompt, prompt)
        if resp.raw_error and not resp.text:
            last_err = f"Meta LLM error: {resp.raw_error}"
            continue
        hyp, code = _extract_hypothesis_and_code(resp.text)
        if not code:
            last_err = "Missing ```python ... ``` block in response."
            continue
        ok, reason = validate_harness_source(code)
        if not ok:
            last_err = f"AST validation failed: {reason}"
            continue
        return hyp or "(no hypothesis provided)", code

    raise RuntimeError(f"Meta-Agent failed to produce a valid harness after {max_retries + 1} tries: {last_err}")

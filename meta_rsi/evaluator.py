"""Dual-view evaluator: In-loop Weak Verifier (+ optional Label-Free Metamorphic Probe) vs Hidden Oracle.

Strict anti-leakage guarantee:
  - In-loop signals visible to Gate & Meta-Agent:
      * dev_low_ok        : whether emitted script matches ground truth on the SINGLE `low` variant
      * dev_low_status    : ok / exception / timeout / no_number / no_code / harness_error
      * dev_low_stderr    : stderr tail on `low`
      * meta_consistent   : (ONLY in metamorphic mode) label-free self-consistency check comparing
                            output(low) vs output(T(low)) — NEVER reads high/zone_edge truth!
      * meta_diag         : label-free explanation of consistency mismatch or crash under T(low)
  - Hidden Oracle signals (logged ONLY for research analysis / plotting; NEVER shown to Meta-Agent or Gate):
      * oracle_category   : full 3-variant classification (correct / lucky / silent_wrong / error / no_code)
      * oracle_test_seen / oracle_test_heldout
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.gsb.harness import matches, score as gsb_score

from .llm import LLMClient
from .metamorphic import perturb_workspace
from .sandbox import load_harness_module, parse_final_number, run_harness_on_workspace, run_python_sandbox


def load_task_splits(
    data_root: Path,
    *,
    dev_per_family: int = 1,
    dev_offset: int = 0,
    test_seen_per_family: int = 1,
    test_heldout_per_family: int = 6,
) -> dict[str, list[dict]]:
    """Deterministic stratified splits from data/v1/tasks.jsonl."""
    rows = [json.loads(line) for line in open(data_root / "tasks.jsonl")]
    by_split_fam: dict[tuple[str, str], list[dict]] = defaultdict(list)
    heldout_fams = set()
    train_fams = set()
    for t in rows:
        by_split_fam[(t["split"], t["family"])].append(t)
        if t["split"] == "train":
            train_fams.add(t["family"])
    for t in rows:
        if t["split"] == "test" and t["family"] not in train_fams:
            heldout_fams.add(t["family"])

    dev: list[dict] = []
    test_seen: list[dict] = []
    test_heldout: list[dict] = []

    for fam in sorted(train_fams):
        dev.extend(by_split_fam[("train", fam)][dev_offset : dev_offset + dev_per_family])
        test_seen.extend(by_split_fam[("test", fam)][:test_seen_per_family])
    for fam in sorted(heldout_fams):
        test_heldout.extend(by_split_fam[("test", fam)][:test_heldout_per_family])

    return {"dev": dev, "test_seen": test_seen, "test_heldout": test_heldout}


def _run_metamorphic_check(
    code: str,
    src_dir: Path,
    orig_val: float,
    answer_type: str,
    *,
    timeout: float = 25.0,
    python: str = sys.executable,
    include_domain_hint: bool = False,
    custom_probes: list[tuple[str, object]] | None = None,
    track_caught: bool = False,
) -> tuple[bool, str, dict]:
    """Label-free self-consistency check: run `code` on T(src_dir) and compare to `orig_val`.

    By default (`include_domain_hint=False`), diagnostics are STRICTLY domain-neutral:
    probe names are anonymized (`probe_1`, `probe_2`, ...) and internal transform mode names
    (`local_aeqd_shift`, `crs_storage_flip`, etc.) are NEVER revealed to the Meta-Agent.
    """
    has_gpkg = any(
        p.suffix.lower() in {".gpkg", ".tif", ".tiff"} or p.name == "metadata.json"
        for p in src_dir.iterdir()
        if p.is_file()
    )
    checks: list[tuple[str, dict | object]] = [
        ("shift_and_shuffle", dict(target_lat=58.3, target_lon=6.0, shuffle_rows=True, flip_gpkg_crs=False, seed=42))
    ]
    if has_gpkg:
        checks.append(("crs_storage_flip", dict(target_lat=-56.2, target_lon=12.0, shuffle_rows=True, flip_gpkg_crs=True, seed=99)))
    if custom_probes:
        for pname, pfn in custom_probes:
            checks.append((pname, pfn))

    details: dict = {}
    for idx, (name, spec) in enumerate(checks, 1):
        label = name if include_domain_hint else f"probe_{idx}"
        with tempfile.TemporaryDirectory() as tmp:
            pert_dir = Path(tmp) / "pert"
            if callable(spec):
                pert_dir.mkdir(parents=True, exist_ok=True)
                pmeta = spec(src_dir, pert_dir)
            else:
                pmeta = perturb_workspace(src_dir, pert_dir, **spec)
            res = run_python_sandbox(code, pert_dir, timeout=timeout, python=python, track_caught=track_caught)
        if res["timed_out"]:
            return False, f"METAMORPHIC_TIMEOUT ({label}): script timed out under equivalent input perturbation.", {name: {"meta": pmeta, "status": "timeout"}}
        if res["returncode"] != 0:
            tail = (res["stderr"].strip().splitlines() or [""])[-1][:220]
            return False, f"METAMORPHIC_CRASH ({label}): crashed under equivalent input perturbation: {tail}", {name: {"meta": pmeta, "status": "exception", "stderr": tail}}
        pert_val = parse_final_number(res["stdout"])
        if pert_val is None:
            return False, f"METAMORPHIC_NO_NUMBER ({label}): did not print a valid number under perturbation.", {name: {"meta": pmeta, "status": "no_number"}}
        consistent = matches(pert_val, orig_val, answer_type)
        details[name] = {"meta": pmeta, "orig": orig_val, "pert": pert_val, "consistent": consistent}
        if not consistent:
            ratio = (pert_val / orig_val) if abs(orig_val) > 1e-12 else float("inf")
            mode = pmeta.get("mode", "") if isinstance(pmeta, dict) else ""
            tag = f"{label}, {mode}" if include_domain_hint else label
            caught_note = ""
            if track_caught and res.get("caught_exceptions"):
                caught_note = f" Caught/swallowed runtime exceptions during probe: {res['caught_exceptions'][:2]}."
            diag = (
                f"METAMORPHIC_INCONSISTENCY ({tag}): "
                f"original output={orig_val:.6g}, equivalent perturbed input output={pert_val:.6g} "
                f"(ratio={ratio:.3g}x, >1% drift). "
                f"The output should remain invariant under this label-preserving input transformation.{caught_note}"
            )
            if include_domain_hint:
                if mode.startswith("tabular"):
                    hint = (
                        "The script likely assumes pre-sorted rows (missing explicit sort_values/tie-breaking), "
                        "relies on positional column indexing, or uses indiscriminate df.dropna() across unrelated columns."
                    )
                else:
                    hint = (
                        "The script likely uses a distorted projection (e.g. EPSG:3857), hardcoded UTM zone, "
                        "degree-based geometry/NN, swapped lat/lon, or ignores layer CRS."
                    )
                diag = f"{diag} {hint}"
            return False, diag, details
    return True, "consistent", details


LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict scientific software verification reviewer auditing Python data-analysis scripts "
    "that already produced a plausible numeric output on one validation input.\n"
    "Determine whether the script implements a general, mathematically sound computation, or whether "
    "it relies on fragile assumptions, distorted coordinate/unit approximations, hardcoded constants, "
    "ignored file metadata/missing-value sentinels, or brittle dataset-specific shortcuts that could "
    "silently fail on equivalent inputs.\n"
    "Respond ONLY with a single JSON object: "
    "{\"verdict\": \"PASS\" or \"FAIL\", \"reason\": \"<1 concise sentence>\"}"
)


def _run_llm_judge_check(
    judge_llm: LLMClient,
    task_prompt: str,
    src_dir: Path,
    code: str,
) -> tuple[bool, str]:
    """Static LLM-as-a-Judge rubric check (baseline alternative to execution-based metamorphic verification)."""
    files = sorted(p.name for p in src_dir.iterdir() if p.is_file())
    user_msg = (
        f"## Task Prompt\n{task_prompt}\n\n"
        f"## Workspace Files\n{', '.join(files)}\n\n"
        f"## Candidate Python Script\n```python\n{code[:3500]}\n```\n\n"
        "Judge whether this script is methodologically sound and free of silent-failure shortcuts. "
        "Output ONLY valid JSON: {\"verdict\": \"PASS\"|\"FAIL\", \"reason\": \"...\"}"
    )
    resp = judge_llm.complete(LLM_JUDGE_SYSTEM_PROMPT, user_msg, temperature=0.0, max_tokens=256)
    if resp.raw_error and not resp.text:
        return True, f"LLM_JUDGE_ERROR_FALLBACK: {resp.raw_error[:120]}"
    raw = (resp.text or "").strip()
    for end in ("</think>", "<channel|>"):
        if end in raw:
            raw = raw.rsplit(end, 1)[1].strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            verdict = str(obj.get("verdict", "")).strip().upper()
            reason = str(obj.get("reason", "")).strip()[:240]
            if verdict == "FAIL":
                return False, f"LLM_JUDGE_REJECT: {reason or 'flagged fragile assumption or shortcut'}"
            if verdict == "PASS":
                return True, f"LLM_JUDGE_PASS: {reason or 'ok'}"
        except json.JSONDecodeError:
            pass
    upper = raw.upper()
    if "FAIL" in upper and "PASS" not in upper:
        return False, f"LLM_JUDGE_REJECT: {raw[:200]}"
    return True, f"LLM_JUDGE_PASS: {raw[:200]}"


def evaluate_harness(
    harness_path: Path,
    tasks: list[dict],
    data_root: Path,
    task_llm: LLMClient,
    *,
    check_metamorphic: bool = False,
    include_domain_hint: bool = False,
    custom_probes: list[tuple[str, object]] | None = None,
    use_oracle_in_loop: bool = False,
    judge_llm: LLMClient | None = None,
    track_caught: bool = False,
    compute_oracle: bool = True,
    skip_dev_exec: bool = False,
    max_llm_calls: int = 4,
    max_exec_calls: int = 12,
    exec_timeout: float = 25.0,
    workers: int = 4,
    python: str = sys.executable,
) -> dict:
    """Evaluate one candidate harness on a list of tasks."""
    harness_mod = load_harness_module(harness_path)
    need_oracle = compute_oracle or use_oracle_in_loop

    def llm_fn(system: str, user: str) -> str:
        r = task_llm.complete(system, user)
        if r.raw_error and not r.text:
            raise RuntimeError(f"LLM backend error: {r.raw_error}")
        return r.text

    def eval_one(task: dict) -> dict:
        low_info = task["variants"]["low"]
        low_dir = data_root / low_info["dir"]
        run = run_harness_on_workspace(
            harness_mod,
            task["prompt"],
            low_dir,
            llm_fn,
            max_llm_calls=max_llm_calls,
            max_exec_calls=max_exec_calls,
            exec_timeout=exec_timeout,
            python=python,
        )
        code = run["code"]
        rec: dict = {
            "task_id": task["task_id"],
            "family": task["family"],
            "answer": task["answer"],
            "llm_calls": run["llm_calls"],
            "elapsed_s": run["elapsed_s"],
            "harness_error": run["harness_error"],
            "code": code,
            "dev_low_status": "no_code",
            "dev_low_ok": False,
            "dev_low_value": None,
            "dev_low_truth": low_info["truth"],
            "dev_low_stderr": "",
            "dev_caught_exceptions": [],
            "meta_checked": False,
            "meta_consistent": False,
            "meta_diag": "",
            "judge_checked": False,
            "judge_pass": False,
            "judge_diag": "",
            "in_loop_pass": False,
            "oracle_category": None,
            "oracle_correct": None,
        }

        if run["harness_error"]:
            rec["dev_low_status"] = "harness_error"
            rec["dev_low_stderr"] = run["harness_error"]
        elif code is not None and not skip_dev_exec:
            r_low = run_python_sandbox(code, low_dir, timeout=exec_timeout, python=python, track_caught=track_caught)
            if track_caught:
                rec["dev_caught_exceptions"] = r_low.get("caught_exceptions", [])
            if r_low["timed_out"]:
                rec["dev_low_status"] = "timeout"
                rec["dev_low_stderr"] = r_low["stderr"][-300:]
            elif r_low["returncode"] != 0:
                rec["dev_low_status"] = "exception"
                rec["dev_low_stderr"] = (r_low["stderr"].strip().splitlines() or [""])[-1][:300]
            else:
                val = parse_final_number(r_low["stdout"])
                if val is None:
                    rec["dev_low_status"] = "no_number"
                    rec["dev_low_stderr"] = f"stdout tail: {r_low['stdout'][-200:]}"
                else:
                    rec["dev_low_status"] = "ok"
                    rec["dev_low_value"] = val
                    rec["dev_low_ok"] = bool(matches(val, low_info["truth"], task["answer"]))

            # Label-free metamorphic check (ONLY runs when dev_low passed and mode requests it)
            if check_metamorphic and rec["dev_low_ok"] and rec["dev_low_value"] is not None:
                rec["meta_checked"] = True
                m_ok, m_diag, _ = _run_metamorphic_check(
                    code,
                    low_dir,
                    rec["dev_low_value"],
                    task["answer"],
                    timeout=exec_timeout,
                    python=python,
                    include_domain_hint=include_domain_hint,
                    custom_probes=custom_probes,
                    track_caught=track_caught,
                )
                rec["meta_consistent"] = m_ok
                rec["meta_diag"] = m_diag

            if judge_llm is not None and rec["dev_low_ok"] and code is not None:
                rec["judge_checked"] = True
                j_ok, j_diag = _run_llm_judge_check(judge_llm, task["prompt"], low_dir, code)
                rec["judge_pass"] = j_ok
                rec["judge_diag"] = j_diag

        if need_oracle:
            sc = gsb_score(task, code, data_root, timeout=exec_timeout, python=python)
            rec["oracle_category"] = sc["category"]
            rec["oracle_correct"] = sc["category"] == "correct"

        if use_oracle_in_loop:
            rec["in_loop_pass"] = bool(rec["oracle_correct"])
        elif judge_llm is not None:
            rec["in_loop_pass"] = bool(rec["dev_low_ok"] and rec["judge_pass"])
        elif check_metamorphic:
            rec["in_loop_pass"] = bool(rec["dev_low_ok"] and rec["meta_consistent"])
        else:
            rec["in_loop_pass"] = bool(rec["dev_low_ok"])

        return rec

    if workers <= 1:
        records = [eval_one(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(eval_one, tasks))

    n = max(len(records), 1)
    oracle_cats = Counter(r["oracle_category"] for r in records if r["oracle_category"] is not None)
    dev_low_pass_n = sum(1 for r in records if r["dev_low_ok"])
    meta_pass_n = sum(1 for r in records if r["dev_low_ok"] and r["meta_consistent"])
    judge_pass_n = sum(1 for r in records if r["dev_low_ok"] and r["judge_pass"])
    in_loop_pass_n = sum(1 for r in records if r["in_loop_pass"])
    dev_err_n = sum(1 for r in records if r["dev_low_status"] != "ok")
    dev_ok_n = sum(1 for r in records if r["dev_low_status"] == "ok")
    dev_swallowed_n = sum(1 for r in records if r["dev_low_status"] == "ok" and r["dev_caught_exceptions"])

    summary = {
        "n_tasks": len(records),
        "dev_low_rate": round(dev_low_pass_n / n, 4),
        "meta_pass_rate": round(meta_pass_n / n, 4) if check_metamorphic else None,
        "judge_pass_rate": round(judge_pass_n / n, 4) if judge_llm is not None else None,
        "in_loop_rate": round(in_loop_pass_n / n, 4),
        "dev_error_rate": round(dev_err_n / n, 4),
        "dev_swallowed_rate": round(dev_swallowed_n / max(dev_ok_n, 1), 4) if track_caught else None,
        "avg_llm_calls": round(sum(r["llm_calls"] for r in records) / n, 3),
        "solved_ids_dev_low": sorted(r["task_id"] for r in records if r["dev_low_ok"]),
        "solved_ids_in_loop": sorted(r["task_id"] for r in records if r["in_loop_pass"]),
    }
    if need_oracle:
        summary.update(
            {
                "oracle_correct_rate": round(oracle_cats["correct"] / n, 4),
                "oracle_lucky_rate": round(oracle_cats["lucky"] / n, 4),
                "oracle_silent_wrong_rate": round(oracle_cats["silent_wrong"] / n, 4),
                "oracle_error_rate": round((oracle_cats["error"] + oracle_cats["no_code"]) / n, 4),
            }
        )

    return {"summary": summary, "records": records}

"""Dual-G4 Split Two-Timescale RSI Trainer: SFT Warm-Start + Online GRPO with Live vLLM LoRA Sync.

Designed to split large models (e.g. `google/gemma-4-26B-A4B-it`) cleanly across two 96 GB Colab G4 GPUs
without cross-NAT NCCL:
  - Inference GPU (`g4-b`): serves frozen base weights via vLLM (`--enable-lora`) + `lora_sync.py` sidecar.
  - Trainer GPU (`g4-a`): holds a single bf16 base copy + dual lightweight PEFT adapters (`sft_ref` frozen
    for exact KL anchoring, `policy` trainable) and pushes bf16 LoRA deltas (~37 MB) to `g4-b` for 43 ms hot-reload.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.gsb.harness import matches

from ..evaluator import _run_metamorphic_check
from ..harnesses.seed import SYSTEM_PROMPT as SEED_SYSTEM_PROMPT
from ..sandbox import parse_final_number, run_python_sandbox
from .collect_trajectories import load_probes_from_dir, select_train_slice
from .lora_sync import push_lora_adapter


def extract_code_block(text: str) -> str | None:
    final = text or ""
    for end in ("</think>", "<channel|>"):
        if end in final:
            final = final.rsplit(end, 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", final, re.S)
    return blocks[-1].strip() if blocks else None


def compute_group_advantages(rewards: list[float], group_size: int, eps: float = 1e-4) -> list[float]:
    """Compute GRPO group-relative normalized advantages; zero-variance groups yield 0.0."""
    if group_size <= 0 or len(rewards) % group_size != 0:
        raise ValueError(f"len(rewards)={len(rewards)} must be divisible by group_size={group_size}")
    advs: list[float] = []
    for i in range(0, len(rewards), group_size):
        grp = rewards[i : i + group_size]
        mean = sum(grp) / group_size
        var = sum((r - mean) ** 2 for r in grp) / group_size
        std = math.sqrt(var)
        if std < 1e-6:
            advs.extend([0.0] * group_size)
        else:
            advs.extend([(r - mean) / (std + eps) for r in grp])
    return advs


def build_masked_sequence(
    prompt_ids: list[int], comp_ids: list[int], max_len: int
) -> tuple[list[int], list[int]]:
    """Concatenate prompt + completion token IDs and mask prompt tokens with -100."""
    if len(prompt_ids) >= max_len - 1:
        p = prompt_ids[-(max_len - 2) :]
        c = comp_ids[:2]
    else:
        p = prompt_ids
        c = comp_ids[: max_len - len(p)]
    input_ids = p + c
    labels = [-100] * len(p) + c
    return input_ids, labels


def score_completion(
    comp_text: str,
    task: dict,
    data_root: Path,
    *,
    reward_mode: str = "metamorphic",
    custom_probes: list[tuple[str, object]] | None = None,
    timeout: float = 15.0,
) -> tuple[float, str, bool]:
    """Score a single completion on `low` only (`weak`) or `low` + label-free metamorphic invariance."""
    code = extract_code_block(comp_text)
    if not code:
        return 0.0, "no_code", False
    uses_3857 = "3857" in code
    low_info = task["variants"]["low"]
    low_dir = data_root / low_info["dir"]
    r = run_python_sandbox(code, low_dir, timeout=timeout)
    if r["timed_out"] or r["returncode"] != 0:
        return 0.0, "err", uses_3857
    val = parse_final_number(r["stdout"])
    if val is None:
        return 0.05, "no_num", uses_3857
    low_ok = bool(matches(val, low_info["truth"], task["answer"]))
    if not low_ok:
        return 0.1, "wrong", uses_3857
    if reward_mode == "weak":
        return 1.0, "low_ok", uses_3857
    m_ok, _, _ = _run_metamorphic_check(
        code,
        low_dir,
        val,
        task["answer"],
        timeout=timeout,
        include_domain_hint=False,
        custom_probes=custom_probes,
    )
    if m_ok:
        return 1.0, "meta_ok", uses_3857
    return 0.05, "meta_fail_shortcut", uses_3857


def discover_lm_lora_targets(model) -> list[str]:
    """Discover language-model-only linear layers (skipping multimodal vision/audio/router modules)."""
    import torch

    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lm_targets = []
    for name, mod in model.named_modules():
        if (
            isinstance(mod, torch.nn.Linear)
            and "vision" not in name
            and "audio" not in name
            and "router" not in name
            and name.endswith(suffixes)
        ):
            lm_targets.append(name)
    attn_targets = [n for n in lm_targets if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
    return attn_targets if len(lm_targets) > 400 else lm_targets


def _resolve_adapter_dir(root: Path, adapter_name: str) -> Path:
    sub = root / adapter_name
    if (sub / "adapter_config.json").exists():
        return sub
    if (root / "adapter_config.json").exists():
        return root
    raise FileNotFoundError(f"Could not locate adapter_config.json under {root} (adapter={adapter_name})")


def _save_single_adapter(model, out_dir: Path, adapter_name: str = "policy") -> Path:
    """Save one named PEFT adapter cleanly into `out_dir` with `adapter_config.json` at top level."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir = out_dir.parent / f".{out_dir.name}_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    model.save_pretrained(tmp_dir, selected_adapters=[adapter_name])
    resolved = _resolve_adapter_dir(tmp_dir, adapter_name)
    shutil.move(str(resolved), str(out_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_dir


def _format_prompt_ids(tokenizer, system: str, user: str) -> list[int]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(prompt_text, add_special_tokens=False)["input_ids"]


def _sequence_token_logps(model, input_ids, labels):
    """Compute per-token log-probabilities for unmasked completion tokens in float32."""
    import torch
    import torch.nn.functional as F

    out = model(input_ids=input_ids, use_cache=False)
    shift_logits = out.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    tok_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return tok_logps, mask


def _generate_vllm_group(
    vllm_url: str,
    model_name: str,
    system: str,
    user: str,
    *,
    n: int = 8,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    timeout: float = 180.0,
) -> list[str]:
    body = json.dumps(
        {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{vllm_url.rstrip('/')}/chat/completions",
        body,
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return [(ch["message"].get("content") or "") for ch in data["choices"]]


def run_sft_stage(
    model,
    tokenizer,
    sft_jsonl: Path,
    out_dir: Path,
    *,
    epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 16,
    max_seq_len: int = 1536,
    seed: int = 42,
    sync_url: str | None = None,
    lora_name: str = "sft_meta",
    unload_prev: str | None = None,
    adapter_key: str = "policy",
) -> dict:
    import torch
    import torch.nn.functional as F

    rows = [json.loads(ln) for ln in open(sft_jsonl)]
    eos = tokenizer.eos_token or ""
    samples = []
    skipped = 0
    for r in rows:
        p_ids = _format_prompt_ids(tokenizer, r["system"], r["user"])
        c_ids = tokenizer(r["assistant"] + eos, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > max_seq_len:
            skipped += 1
            continue
        inp, lab = build_masked_sequence(p_ids, c_ids, max_seq_len)
        samples.append((inp, lab))
    if not samples:
        raise RuntimeError(f"No valid SFT samples under max_seq_len={max_seq_len} from {sft_jsonl}")

    print(
        f"[sft:{lora_name}] Loaded {len(samples)} samples (skipped {skipped} > {max_seq_len} tok), "
        f"epochs={epochs}, bs={batch_size}, lr={lr}",
        flush=True,
    )
    model.set_adapter(adapter_key)
    model.train()
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad and adapter_key in n]
    if not trainable_params:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_steps = max(1, math.ceil(len(samples) / batch_size) * epochs)
    warmup = max(1, int(0.1 * total_steps))
    rng = random.Random(seed)
    step = 0
    loss_history = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    for ep in range(epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            step += 1
            # Linear warmup + cosine decay
            if step <= warmup:
                scale = step / warmup
            else:
                prog = (step - warmup) / max(1, total_steps - warmup)
                scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            for pg in opt.param_groups:
                pg["lr"] = lr * scale

            opt.zero_grad(set_to_none=True)
            batch_loss_val = 0.0
            for idx in batch_idx:
                inp, lab = samples[idx]
                inp_t = torch.tensor([inp], dtype=torch.long, device="cuda")
                lab_t = torch.tensor([lab], dtype=torch.long, device="cuda")
                tok_logps, mask = _sequence_token_logps(model, inp_t, lab_t)
                m = mask.float()
                nll = -(tok_logps * m).sum() / m.sum().clamp(min=1.0)
                (nll / len(batch_idx)).backward()
                batch_loss_val += float(nll.detach().item()) / len(batch_idx)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            loss_history.append({"step": step, "epoch": ep + 1, "loss": round(batch_loss_val, 4), "lr": round(lr * scale, 7)})
            if step == 1 or step % 5 == 0 or step == total_steps:
                print(
                    f"[sft:{lora_name}] ep={ep+1}/{epochs} step={step}/{total_steps} "
                    f"loss={batch_loss_val:.4f} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    sft_dir = _save_single_adapter(model, out_dir / lora_name, adapter_name=adapter_key)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    sync_info = None
    if sync_url:
        sync_info = push_lora_adapter(
            sft_dir, sync_url=sync_url, lora_name=lora_name, unload_prev=unload_prev, cast_bf16=True
        )
        print(f"[sft:{lora_name}] Synced to vLLM: {sync_info}", flush=True)

    summary = {
        "stage": "sft",
        "lora_name": lora_name,
        "sft_jsonl": str(sft_jsonl),
        "adapter_dir": str(sft_dir),
        "n_samples": len(samples),
        "skipped": skipped,
        "epochs": epochs,
        "steps": step,
        "train_s": round(time.time() - t0, 1),
        "peak_vram_gb": peak_gb,
        "initial_loss": loss_history[0]["loss"],
        "final_loss": loss_history[-1]["loss"],
        "sync": sync_info,
        "loss_history": loss_history,
    }
    (out_dir / f"{lora_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_grpo_stage(
    model,
    tokenizer,
    *,
    sft_adapter_dir: Path,
    data_root: Path,
    out_dir: Path,
    vllm_url: str,
    sync_url: str,
    initial_vllm_model: str,
    lora_prefix: str = "grpo_meta",
    reward_mode: str = "metamorphic",
    custom_probes: list[tuple[str, object]] | None = None,
    train_offset: int = 20,
    train_per_family: int = 35,
    max_steps: int = 30,
    tasks_per_step: int = 4,
    num_generations: int = 8,
    max_completion_tokens: int = 1024,
    max_seq_len: int = 1536,
    lr: float = 1e-5,
    beta: float = 0.02,
    clip_eps: float = 0.2,
    sync_every: int = 2,
    score_workers: int = 16,
    exec_timeout: float = 25.0,
    seed: int = 42,
) -> dict:
    import torch

    # Load frozen SFT anchor alongside trainable policy (only +74 MB VRAM!)
    if "sft_ref" not in model.peft_config:
        model.load_adapter(str(sft_adapter_dir), adapter_name="sft_ref", is_trainable=False)
    for name, p in model.named_parameters():
        if "sft_ref" in name:
            p.requires_grad_(False)
    model.set_adapter("policy")
    for name, p in model.named_parameters():
        if "sft_ref" in name:
            p.requires_grad_(False)
        elif ".policy." in name:
            p.requires_grad_(True)
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad and "sft_ref" not in n]
    assert trainable_params, "No trainable policy parameters found for GRPO"
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

    train_tasks = select_train_slice(data_root, train_offset=train_offset, train_per_family=train_per_family)
    rng = random.Random(seed)
    eos = tokenizer.eos_token or ""
    active_vllm_model = initial_vllm_model
    prev_synced_temp: str | None = None

    step_logs: list[dict] = []
    t_all = time.time()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, max_steps + 1):
        batch_tasks = rng.sample(train_tasks, k=min(tasks_per_step, len(train_tasks)))

        # 1. Parallel remote rollout generation via inference GPU vLLM
        t_gen = time.time()
        with ThreadPoolExecutor(max_workers=len(batch_tasks)) as ex:
            futs = [
                ex.submit(
                    _generate_vllm_group,
                    vllm_url,
                    active_vllm_model,
                    SEED_SYSTEM_PROMPT,
                    t["prompt"],
                    n=num_generations,
                    temperature=0.8,
                    max_tokens=max_completion_tokens,
                )
                for t in batch_tasks
            ]
            groups = [f.result() for f in futs]
        gen_s = round(time.time() - t_gen, 2)

        flat_pairs = [(t, comp) for t, comps in zip(batch_tasks, groups) for comp in comps]

        # 2. Parallel sandbox + metamorphic scoring on trainer VM's 48 CPU cores
        t_sc = time.time()
        with ThreadPoolExecutor(max_workers=score_workers) as ex:
            scored = list(
                ex.map(
                    lambda pair: score_completion(
                        pair[1],
                        pair[0],
                        data_root,
                        reward_mode=reward_mode,
                        custom_probes=custom_probes,
                        timeout=exec_timeout,
                    ),
                    flat_pairs,
                )
            )
        score_s = round(time.time() - t_sc, 2)

        rewards = [x[0] for x in scored]
        cats = [x[1] for x in scored]
        n3857 = sum(1 for x in scored if x[2])
        n_full = sum(1 for r in rewards if r >= 0.99)
        n_blocked = sum(1 for c in cats if c == "meta_fail_shortcut")
        advs = compute_group_advantages(rewards, num_generations)

        # Keep only rollouts with non-zero advantage
        active_items = []
        for (task, comp), adv in zip(flat_pairs, advs):
            if abs(adv) < 1e-8 or not comp.strip():
                continue
            p_ids = _format_prompt_ids(tokenizer, SEED_SYSTEM_PROMPT, task["prompt"])
            c_ids = tokenizer(comp + eos, add_special_tokens=False)["input_ids"]
            if len(c_ids) < 2:
                continue
            inp, lab = build_masked_sequence(p_ids, c_ids, max_seq_len)
            active_items.append((inp, lab, adv))

        # 3. Exact reference & old policy logprobs + clipped GRPO + Schulman k3 KL backward
        t_tr = time.time()
        mean_kl = 0.0
        mean_loss = 0.0
        if active_items:
            ref_logps_list = []
            old_logps_list = []
            model.eval()
            with torch.no_grad():
                model.set_adapter("sft_ref")
                for inp, lab, _ in active_items:
                    inp_t = torch.tensor([inp], dtype=torch.long, device="cuda")
                    lab_t = torch.tensor([lab], dtype=torch.long, device="cuda")
                    rlp, _ = _sequence_token_logps(model, inp_t, lab_t)
                    ref_logps_list.append(rlp.detach())
                model.set_adapter("policy")
                for inp, lab, _ in active_items:
                    inp_t = torch.tensor([inp], dtype=torch.long, device="cuda")
                    lab_t = torch.tensor([lab], dtype=torch.long, device="cuda")
                    olp, _ = _sequence_token_logps(model, inp_t, lab_t)
                    old_logps_list.append(olp.detach())

            model.set_adapter("policy")
            model.train()
            opt.zero_grad(set_to_none=True)
            denom = float(len(active_items))
            for (inp, lab, adv), ref_lp, old_lp in zip(active_items, ref_logps_list, old_logps_list):
                inp_t = torch.tensor([inp], dtype=torch.long, device="cuda")
                lab_t = torch.tensor([lab], dtype=torch.long, device="cuda")
                cur_lp, mask = _sequence_token_logps(model, inp_t, lab_t)
                m = mask.float()
                n_tok = m.sum().clamp(min=1.0)
                ratio = torch.exp((cur_lp - old_lp).clamp(-10.0, 10.0))
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                pg_loss = -torch.min(surr1, surr2)
                log_ref = (ref_lp - cur_lp).clamp(-10.0, 10.0)
                kl = torch.exp(log_ref) - log_ref - 1.0
                loss = ((pg_loss + beta * kl) * m).sum() / n_tok
                (loss / denom).backward()
                mean_kl += float(((kl * m).sum() / n_tok).detach().item()) / denom
                mean_loss += float(loss.detach().item()) / denom
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
        train_s = round(time.time() - t_tr, 2)

        # 4. Hot-reload updated policy LoRA onto inference GPU vLLM
        sync_s = 0.0
        if active_items and (step % sync_every == 0 or step == max_steps):
            t_sy = time.time()
            step_name = f"{lora_prefix}_s{step:03d}"
            step_dir = _save_single_adapter(model, out_dir / "latest_policy", adapter_name="policy")
            sres = push_lora_adapter(
                step_dir,
                sync_url=sync_url,
                lora_name=step_name,
                unload_prev=prev_synced_temp,
                cast_bf16=True,
            )
            prev_synced_temp = step_name
            active_vllm_model = step_name
            sync_s = round(time.time() - t_sy, 2)
            print(
                f"[grpo:{lora_prefix}] Synced {step_name} in {sync_s}s "
                f"(payload={sres.get('payload_mb')} MB, vllm_load={sres.get('load_ms')} ms)",
                flush=True,
            )

        rec = {
            "step": step,
            "vllm_model": active_vllm_model,
            "mean_r": round(sum(rewards) / max(len(rewards), 1), 4),
            "n_full": n_full,
            "n_meta_blocked": n_blocked,
            "n_3857": n3857,
            "n_rollouts": len(rewards),
            "n_active": len(active_items),
            "loss": round(mean_loss, 5),
            "kl": round(mean_kl, 5),
            "gen_s": gen_s,
            "score_s": score_s,
            "train_s": train_s,
            "sync_s": sync_s,
        }
        step_logs.append(rec)
        print(
            f"[grpo:{lora_prefix}] step={step:2d}/{max_steps} mean_r={rec['mean_r']:.3f} "
            f"full={n_full:2d}/{len(rewards)} blocked={n_blocked:2d} 3857={n3857:2d} "
            f"act={len(active_items):2d} kl={mean_kl:.4f} "
            f"(gen={gen_s}s sc={score_s}s tr={train_s}s sy={sync_s}s)",
            flush=True,
        )

    final_name = f"{lora_prefix}_final"
    final_dir = _save_single_adapter(model, out_dir / final_name, adapter_name="policy")
    final_sync = push_lora_adapter(
        final_dir,
        sync_url=sync_url,
        lora_name=final_name,
        unload_prev=prev_synced_temp,
        cast_bf16=True,
    )
    summary = {
        "stage": "grpo",
        "lora_name": final_name,
        "adapter_dir": str(final_dir),
        "reward_mode": reward_mode,
        "max_steps": max_steps,
        "train_s": round(time.time() - t_all, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "final_sync": final_sync,
        "steps": step_logs,
    }
    (out_dir / f"{final_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-G4 Split Two-Timescale SFT + GRPO Trainer")
    ap.add_argument("--mode", required=True, choices=["sft", "grpo", "sft_and_grpo"])
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data/v1")
    ap.add_argument("--sft-jsonl", default="")
    ap.add_argument(
        "--extra-sft",
        action="append",
        default=[],
        help="Optional additional independent SFT adapters trained on the same loaded base model (format: name=path.jsonl)",
    )
    ap.add_argument("--sft-lora-dir", default="", help="Existing SFT adapter dir when --mode grpo")
    ap.add_argument("--sft-lora-name", default="sft_meta")
    ap.add_argument("--sft-epochs", type=int, default=2)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--sft-batch-size", type=int, default=16)
    ap.add_argument("--grpo-prefix", default="grpo_meta")
    ap.add_argument("--reward-mode", default="metamorphic", choices=["metamorphic", "weak"])
    ap.add_argument("--grpo-steps", type=int, default=30)
    ap.add_argument("--grpo-lr", type=float, default=1e-5)
    ap.add_argument("--grpo-beta", type=float, default=0.02)
    ap.add_argument("--tasks-per-step", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--sync-every", type=int, default=2)
    ap.add_argument("--train-offset", type=int, default=20)
    ap.add_argument("--train-per-family", type=int, default=35)
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--sync-url", default="http://127.0.0.1:8003")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--max-completion-tokens", type=int, default=1024)
    ap.add_argument("--probe-dir", default="", help="Optional dir of certified metamorphic probe .py files")
    ap.add_argument("--exec-timeout", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Periodic heartbeat so remote CLI executors never hit silence timeouts
    stop_hb = threading.Event()
    def _hb():
        t0 = time.time()
        while not stop_hb.wait(15.0):
            print(f"[split-2ts hb] elapsed={time.time() - t0:.0f}s", flush=True)
    threading.Thread(target=_hb, daemon=True).start()

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t_load = time.time()
        print(f"[split-2ts] Loading base model {a.model} in bf16...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(a.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()

        if a.mode in {"sft", "sft_and_grpo"}:
            targets = discover_lm_lora_targets(base)
            print(f"[split-2ts] Discovered {len(targets)} LM LoRA targets (sample: {targets[:3]})", flush=True)
            cfg = LoraConfig(
                r=a.lora_r,
                lora_alpha=a.lora_alpha,
                lora_dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, cfg, adapter_name="policy")
            model.print_trainable_parameters()
            print(f"[split-2ts] Loaded base + LoRA in {time.time() - t_load:.1f}s", flush=True)

            # Train any extra independent control SFT adapters first on the same loaded base weights
            for spec in a.extra_sft:
                ctrl_name, ctrl_path = spec.split("=", 1)
                model.add_adapter("policy", cfg) if "policy" not in model.peft_config else None
                # Reset a fresh adapter for this control
                model.add_adapter(ctrl_name, cfg)
                model.set_adapter(ctrl_name)
                # Temporarily run SFT on ctrl_name by swapping adapter name in run_sft_stage
                _run_named_sft = run_sft_stage(
                    model,
                    tokenizer,
                    Path(ctrl_path),
                    out_dir,
                    epochs=a.sft_epochs,
                    lr=a.sft_lr,
                    batch_size=a.sft_batch_size,
                    max_seq_len=a.max_seq_len,
                    seed=a.seed,
                    sync_url=a.sync_url,
                    lora_name=ctrl_name,
                    adapter_key=ctrl_name,
                )
                model.delete_adapter(ctrl_name)
                model.set_adapter("policy")

            sft_res = run_sft_stage(
                model,
                tokenizer,
                Path(a.sft_jsonl),
                out_dir,
                epochs=a.sft_epochs,
                lr=a.sft_lr,
                batch_size=a.sft_batch_size,
                max_seq_len=a.max_seq_len,
                seed=a.seed,
                sync_url=a.sync_url,
                lora_name=a.sft_lora_name,
                adapter_key="policy",
            )
            sft_dir = Path(sft_res["adapter_dir"])
        else:
            sft_dir = Path(a.sft_lora_dir)
            model = PeftModel.from_pretrained(base, str(sft_dir), adapter_name="policy", is_trainable=True)
            model.print_trainable_parameters()
            print(f"[split-2ts] Loaded base + existing SFT adapter in {time.time() - t_load:.1f}s", flush=True)

        if a.mode in {"grpo", "sft_and_grpo"}:
            probes = (
                load_probes_from_dir(Path(a.probe_dir), out_dir / "_probe_mods")
                if a.probe_dir
                else None
            )
            run_grpo_stage(
                model,
                tokenizer,
                sft_adapter_dir=sft_dir,
                data_root=Path(a.data_root),
                out_dir=out_dir,
                vllm_url=a.vllm_url,
                sync_url=a.sync_url,
                initial_vllm_model=a.sft_lora_name,
                lora_prefix=a.grpo_prefix,
                reward_mode=a.reward_mode,
                custom_probes=probes,
                train_offset=a.train_offset,
                train_per_family=a.train_per_family,
                max_steps=a.grpo_steps,
                tasks_per_step=a.tasks_per_step,
                num_generations=a.num_generations,
                max_completion_tokens=a.max_completion_tokens,
                max_seq_len=a.max_seq_len,
                lr=a.grpo_lr,
                beta=a.grpo_beta,
                sync_every=a.sync_every,
                exec_timeout=a.exec_timeout,
                seed=a.seed,
            )
    finally:
        stop_hb.set()


if __name__ == "__main__":
    main()

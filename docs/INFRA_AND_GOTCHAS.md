# Compute Inventory, Budgets & Engineering Runbook

> Long-term infrastructure reference for local NixOS development, Colab G4 inference/training, API usage, and hard-won gotchas.
> Last updated: 2026-11-21.

---

## 1. Hardware & Cloud Inventory

### Local Workstation
- **OS**: NixOS Linux (`shell.nix` provides reproducible Python 3.14.10 + `geopandas`, `pyproj`, `shapely`, `numpy`, `pandas`, `pyogrio`, `scipy`, `pytest`).
- **Specs**: RTX 4060 Laptop (8 GB VRAM), 20 CPU threads, 31 GB RAM.
- **Command pattern**: Run all local geospatial Python commands through `nix-shell --run "..."`.

### Cloud GPU — Google Colab G4
- **CLI**: `colab` authenticated via `oauth2`.
- **Hardware (verified 2026-11-18)**: NVIDIA RTX PRO 6000 Blackwell Server Edition (**96 GB / 97,887 MiB VRAM**, bf16), 48 CPU cores, 176 GB RAM, 236 GB disk, PyTorch 2.14.1 + CUDA 13.0.
- **Quota & Concurrency**: ~1,000 compute units/month at **10.05 CU/hr** (~99 hours/month). Account supports **2 concurrent G4 sessions** (`colab new -s g4-a --gpu G4` + `colab new -s g4-b --gpu G4`).
  - **Default for batch experiments**: Always spin up **both G4 sessions** and shard multi-seed evolutions or multi-harness evaluations 50/50 across them — total CU cost is identical to serial execution while cutting wall-clock time in half.
  - **Prefer on-VM execution for evaluations**: Upload a compact tarball (`benchmarks/`, `meta_rsi/`, harnesses, and needed `data/v1` task dirs — all 480 test tasks compress to ~7 MB) and run `eval_test` / `eval_ood` directly on each 48-core VM against `http://127.0.0.1:8000/v1` (2 concurrent eval processes × 24 workers per VM). This eliminates SSH tunnel RTT and the 60-minute WebSocket drop.
- **Cost Discipline**: Stop all sessions (`colab stop -s <name>`) immediately after runs; verify zero idle billing with `colab sessions`.

### API & Cost-Control Architecture
- **`gemini-api` CLI**: Uses local OS keyring authentication (`--meta-backend gemini-cli`); keys stay on the host.
- **Gemini Decisions API (`gemini-decision-1`)**: Outputs calibrated option probabilities (no text generation), up to 1,000 questions/request, ~150–250 ms latency, \$0.10 / 1M input tokens, free output (verified on 7.18M input tokens with 0 failures on 2026-11-18).
- **Zero-Marginal-Cost Architecture**: Run high-frequency Task Agents on Colab G4 `vLLM` (open-weight 4B–27B); reserve external API calls for low-frequency Meta-Agent proposals (1 call/generation) and strong-base validation.

---

## 2. Inference & Harness Evolution Gotchas

1. **NixOS `colab ssh -- CMD` bug**: Direct `colab ssh -- CMD` fails with `ModuleNotFoundError: No module named 'colab_cli'` because `colab` builds a ProxyCommand using bare `python3.14`. Always use explicit SSH forwarding:
   ```bash
   ssh -o "ProxyCommand=colab ssh --proxy-mode -s rsi-g4" \
       -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
       -o ExitOnForwardFailure=yes -o ServerAliveInterval=25 \
       -N -L 127.0.0.1:18000:127.0.0.1:8000 root@colab
   ```
2. **60-minute WebSocket tunnel drop**: Colab SSH WebSocket tunnels disconnect after ~60 minutes while the remote VM and `vLLM` remain alive. Re-open the tunnel, and **always audit `usage.errors == 0`** in every output JSON after long evaluations so dropped calls are never miscounted as task `error`s.
3. **Single-G4 `vLLM` throughput**: `google/gemma-4-E4B-it` uses only **15.2 GiB / 96 GiB**. Launch via `meta_rsi/infra/start_g4_vllm.py` (`--max-num-seqs 128`); one G4 handles 3 concurrent evolution processes via continuous batching, and dual G4 sessions can serve ports `18000` and `18001`.
4. **Meta-Agent (`gemini-4-hydrogen`) thinking truncation**: High thinking effort consumes ~6k thought tokens and truncates generated `harness.py` under `max_tokens=8192`. Always pass `--meta-thinking-level low --meta-max-tokens 16384` (~15 s/call, zero truncation).
5. **Gemma 4 `vLLM` thinking switch & BF16 jitter**: `meta_rsi/llm.py` explicitly passes `chat_template_kwargs: {"enable_thinking": bool}`. Use `--task-temp 0.0` for evaluation; ~1–2% BF16 dynamic-batch jitter still occurs, so always run paired multi-seed experiments.
6. **Code-fence extraction inside evolved harnesses**: Evolved `harness.py` files often contain literal ```` ```python ```` regexes; non-greedy regexes truncate early. `meta_rsi/meta_agent.py` uses line-anchored opening/closing fence matching.
7. **Thinking-span stripping**: When evaluating thinking models (e.g., Qwen3.8, Gemma 4 thinking ON), strip everything before `</think>` or `<channel|>` so draft snippets inside reasoning traces are never scored as final code.
8. **`colab exec` 30-second silence timeout**: `colab exec` disconnects locally if a remote script produces no stdout for 30s (the remote Jupyter kernel keeps running, though). Always pass `--timeout 1800` and emit a periodic heartbeat (`flush=True`) every ~15s in batch drivers; use `colab ls -s <name> <path>` (Contents API) to check remote output files without blocking behind a busy kernel.

---

## 3. Colab G4 GRPO / LoRA Training Benchmarks & Gotchas (For Two-Timescale RSI P1)

Measured on Colab G4 (96 GB) via `gis-grpo-pilot` (`TRL 1.18.1` `GRPOTrainer`, LoRA `r=16`, 4 tasks × 8 samples = 32 completions/step, 2026-11-18):

| Model & Setting | Step Time | Gen Time | Avg Tokens | Peak VRAM | Notes |
|---|---:|---:|---:|---:|---|
| Qwen3-1.7B (no vLLM) | 6–11 s | ~5 s | 125–155 | 9.5 GB | Baseline too weak (4/6 steps all-zero reward) |
| Gemma 4 E4B (think, no vLLM) | ~110 s | ~90 s | 1,180–1,300 | 24.2 GB | — |
| **Gemma 4 E4B (think, vLLM colocate)** | **41–51 s** | **24–29 s** | 1,237–1,311 | **33.1 GB** | **3.5× faster gen; ~3.7 hr (~37 CU) per 300 steps** |
| Qwen3.8-27B (no think, max 1024, no vLLM) | ~126 s | ~96 s | 455–556 | 61.8 GB | 512-token cap caused 25–50% truncation |
| Qwen3.8-27B (think `low`, no vLLM) | 250–335 s | 191–257 s | 934–1,110 | 62.7 GB | Default `xhigh` hits 4096 cap (~650 s/step) |
| Qwen3.8-27B + vLLM colocate (sleep mode) | OOM | — | — | >96 GB | 50 GB training weights + vLLM copy exceeds 96 GB |
| Gemma 4 26B-A4B (think, no vLLM) | 800–940 s | 750–877 s | 2,383–2,779 | 61.4 GB | MoE slow without vLLM |
| **Gemma 4 26B-A4B (Split 2×G4: vLLM + LoRA r=16)** | **~30–68 s** | **~24–62 s** | 500–800 | **67.3 GB** | **Verified 2026-11-21**: `g4-b` vLLM (`--enable-lora`) + `g4-a` bf16 trainer (3.1 s bwd, 74 MB LoRA, **43 ms** live hot-reload); **13×–25× faster** than single-card |
| Gemma 4 E4B (no-think, no vLLM, max 768) | **~41 s** | ~37 s | 250–450 | ~24 GB | Verified 2026-11-21 (60 steps GRPO in ~41 min) |
| Gemma 4 31B (think) | OOM | — | — | >96 GB | OOM on single-sample backward pass |

### Training Environment Gotchas (Verify Before P1 Runs)
- **`colab` CLI non-interactive hangs**: Always pass `--auth=oauth2` and redirect stdin (`</dev/null`) when launching `colab new` / `colab exec` / `colab stop` from background tasks.
- **`VLLM::EngineCore` zombie VRAM leak**: `pkill -f vllm.entrypoints` leaves `VLLM::EngineCore` holding GPU memory. Always kill with `pkill -9 -f 'vllm|VLLM|EngineCore'` before switching models or starting training.
- **`torchao` conflict on Colab**: Preinstalled `torchao 0.10.0` conflicts with `peft`. Always run `pip uninstall -y torchao` before installing `trl` / `peft`.
- **`vllm 0.34.0` + `trl 1.18.2` colocate `cutlass` error**: Single-card `vllm_mode="colocate"` can raise `ModuleNotFoundError: No module named 'cutlass'` under Colab's `vllm 0.34.0`. Either use native bf16 (`use_vllm=False`, ~41 s/step for E4B no-think) or use the **Split 2×G4** architecture (`g4-b` standalone vLLM with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` + `--enable-lora` + `POST /v1/load_lora_adapter`, `43 ms` reload!).
- **Gemma 4 multimodal LoRA module collision**: Gemma 4's vision/audio towers share short layer names (`q_proj`, `k_proj`, etc.) with the language model. Specify **full module paths** targeting only the language-model linear layers (`language_model.*`).
- **Gemma 4 merged checkpoint vLLM reload**: When saving a merged Gemma 4 LoRA checkpoint for vLLM serving, call `AutoProcessor.from_pretrained(MODEL_ID).save_pretrained(merged_dir)` alongside `merged.save_pretrained` and `tokenizer.save_pretrained` — otherwise vLLM fails with `OSError: Can't load feature extractor` (missing `processor_config.json`).
- **Qwen3.8-27B hybrid architecture**: Requires `flash-linear-attention` + `causal-conv1d`, and LoRA targets must include DeltaNet layers (`in_proj_qkv`, `in_proj_z`, `out_proj`) alongside attention/MLP projections.
- **Checkpointing**: Always mirror checkpoints to Google Drive or download immediately to avoid session-drop loss.

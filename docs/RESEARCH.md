# `meta-rsi` — Long-Term Research Plan, Experiment Log & Roadmap

> Living master document for the `meta-rsi` research project, undergraduate thesis, and SCI / top-venue manuscript.
> Repository: `https://github.com/Konakonai/meta-rsi`
> Last updated: 2026-11-21.
> Related long-term docs: [LITERATURE.md](LITERATURE.md) · [GSB_V1_BASELINE.md](GSB_V1_BASELINE.md) · [INFRA_AND_GOTCHAS.md](INFRA_AND_GOTCHAS.md)

---

## 1. Research Scope & Objectives

- **Research Topic**: *Recursive Self-Improvement for Large Language Model Agents in Geospatial Analysis*（面向地理空间分析的大语言模型智能体递归自我改进方法研究）.
- **Core Goal**: Build and rigorously evaluate a self-evolving Agent Harness for executable geospatial analysis that resists weak-verifier **Proxy Lock-In** without extra ground-truth labels, and progress from bounded harness iteration to true cross-generation Recursive Self-Improvement (RSI).
- **Modality Roadmap**: Grounded initially in local/large-extent GeoPandas & PyProj tasks (`benchmarks/gsb`), expanding toward real OpenStreetMap slices, raster/DEM workflows, and **Google Earth Engine (GEE)** tasks in later phases.

---

## 2. Core Research Thesis & Verified Results (v0.2)

### Core Claim
> Self-evolving agent loops exploit weak in-loop verifiers. Under single-location verification, strict regression gating (`Preserve-and-Extend`, DarwinX-style) suffers from **Proxy Lock-In** — freezing early shortcut hacks (e.g., `EPSG:3857` near the equator) and rejecting genuinely general fixes. Adding **Label-Free Metamorphic Invariance Gating** (zero extra labels) breaks Proxy Lock-In, slashes spurious correctness, and avoids the runtime skill contamination caused by hand-written prompt rules.

### v0.2 Experimental Setup (4 Paired Disjoint Seeds)
- **Splits**: 4 disjoint dev sets (`v3_gis_s{0..3}_{naive,preserve,meta}`), 36 dev tasks/seed (`2 × 18` seen families, `dev_offset = 0, 2, 4, 6`, 144 train tasks total), 4 generations of evolution.
- **Task LLM**: Frozen `google/gemma-4-E4B-it` (`enable_thinking=False`, `temp=0.0`, `max_tokens=2048`, hard cap 4 calls/task).
- **Meta LLM**: `gemini-4-hydrogen` (`thinking_level=low`, `temp=0.6`, `max_tokens=16384`).
  - *Audit note*: v0.2 runs used the original `evaluator.py` where `meta_diag` included a short domain hint string on metamorphic mismatch; `evaluator.py` has now been updated to be **strictly domain-neutral by default** (`--meta-domain-hints` opt-in for controlled ablation; see Risk 2 below).
- **Unseen Test Set (`runs/eval480/`, all 13 JSONs verified `usage.errors == 0`)**: **Full 480 unseen 3-location test tasks** (360 seen + 120 held-out = 1,440 variant executions/harness), directly apples-to-apples with all 5 `zero_shot_v1` baselines. (The earlier 150-task stratified subset remains archived under `runs/eval150/`.)
- **Polar OOD Stress Test (`runs/eval_ood/`, all 13 JSONs verified `usage.errors == 0`)**: **Full 440 local unseen test tasks** (320 seen + 120 held-out) relocated to `86.5°N` and `-85.5°S` (disjoint from metamorphic targets `58.3°N / -56.2°S`, where `estimate_utm_crs()` raises `RuntimeError`), evaluated with `--track-caught`.

### Main 4-Seed Aggregate (`mean ± std`, Full `n = 480` Unseen Test Tasks)

Reproducible anytime via `python -m meta_rsi.eval.analysis runs/eval480 --ood-dir runs/eval_ood`:

| Method | Seeds | Dev `Low` (Weak) | Dev `Oracle` (3-loc) | Dev Gap | **Test ALL (`n=480`)** | **Test Seen (`n=360`)** | **Test Held-out (`n=120`)** | **Test `Lucky`** | Test `Err` | Calls | Lock-In (Gap ≥15%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gen-0 Seed (`harnesses/seed.py`) | 1 | 45.8% | 17.6% | +28.2% | 24.2% | 23.1% | 27.5% | 31.2% | 18.1% | 1.00 | — |
| `naive` (hill-climbing) | 4 | 84.7 ± 2.8% | 66.7 ± 24.3% | 18.1 ± 23.6% | 66.9 ± 22.7% | 64.7 ± 18.5% | 73.3 ± 35.7% | 18.6 ± 21.6% | 4.1 ± 1.7% | 1.30 | 1/4 |
| `preserve` (DarwinX-style) | 4 | 75.7 ± 11.9% | 50.7 ± 21.9% | 25.0 ± 12.6% | 52.1 ± 22.4% | 51.0 ± 20.5% | 55.4 ± 29.1% | 24.6 ± 15.4% | 7.4 ± 2.9% | 1.26 | **3/4** |
| **`metamorphic` (Ours)** | 4 | 80.6 ± 2.3% | **77.8 ± 3.9%** | **2.8 ± 5.6%** | **78.4 ± 3.8%** | **73.8 ± 4.9%** | **92.5 ± 5.1%** | **2.1 ± 1.8%** | 8.9 ± 3.0% | 1.50 | **0/4** |

- **Paired seed comparisons (`n = 480`)**: `metamorphic` beats `preserve` on 4/4 seeds (`+26.4 ± 21.4%` Test ALL, paired $t=+2.46, p=0.0905$) and slashes `Lucky` on 4/4 seeds (`-22.4 ± 15.7%`, paired $t=-2.86, p=0.0648$); beats `naive` on `Lucky` on 4/4 seeds (`-16.5 ± 22.5%`) and cuts cross-seed std by **5.9×–6.0×** (`3.8%` vs `22.4%`/`22.7%`).
- **4B No-Think + Evolved Harness vs 26B/27B Thinking (same 480 tasks)**: Best evolved harness (`runs/v3_gis_s2_meta/gen_003/harness.py`, `1.29` calls/task) reaches **81.9% ALL / 99.2% Held-out (119/120) / 1.0% lucky**, surpassing zero-shot thinking `gemma-4-26B-A4B-it` (`71.7%` ALL / `13.4%` lucky) by **+10.2%** and `Qwen3.8-27B` (`62.2%`) by **+19.7%** (see [GSB_V1_BASELINE.md](GSB_V1_BASELINE.md)).

### Task Category Breakdown (Full `n = 480`, `correct% / lucky%`)

| Method | `LOCAL_PROJ` (15 families) | `NON_PROJ_TRAP` (4 families)* | `LARGE_EXTENT` (2 families) |
|---|---:|---:|---:|
| Gen-0 Seed | 18.6% / 37.8% | 42.5% / 10.0% | 37.5% / 15.0% |
| `naive` | 70.6 ± 30.4% / 22.1% | 52.2 ± 9.9% / 9.1% | 62.5 ± 28.1% / 6.2% |
| `preserve` (DarwinX) | 52.8 ± 28.6% / 30.1% | 52.5 ± 12.4% / 6.9% | 45.0 ± 5.4% / 10.6% |
| **`metamorphic` (Ours)** | **87.4 ± 4.3% / 1.7%** | **51.9 ± 8.8% / 1.9%** | 51.2 ± 16.4% / 6.2% |

Key diagnostic insights from the full-480 breakdown:
1. **Zero skill contamination on non-projection tasks**: Unlike hand-written CRS rules (`e4b_think_rules`, which collapsed `points_in_polygon` `78.7% -> 41.9%` and `area_weighted_population` `43.8% -> 8.7%`), `metamorphic` *improves* `NON_PROJ_TRAP` (`42.5% -> 51.9%`, with only `1.9%` lucky). *(\*Audit note: v0.2 grouped `covered_population` inside `NON_PROJ_TRAP`; per `gsb_v1_audit.log`, `covered_population` uses a metric buffer and fails `EPSG:3857` on `166/170` tasks, so `meta_rsi/eval/analysis.py` has been updated to restrict `NON_PROJ_TRAP` to the 3 purely projection-invariant families: `points_in_polygon`, `area_weighted_population`, `mean_per_district`.)*
2. **Principled headroom on `LARGE_EXTENT`**: Our current large-extent isometry (longitude shift + latitude mirror) preserves single-UTM-zone distortion when the UTM zone tracks the shifted centroid, explaining why `LARGE_EXTENT` stays at `51.2%` — motivating **self-discovered / stronger invariance operators** in Phase P1.

### Polar OOD Stress Test (`runs/eval_ood/`, Full `n = 440` Local Tasks, `>84°N / <80°S`)

| Method | Seeds | **Polar ALL (`n=440`)** | **Polar Held-out (`n=120`)** | **Polar `Lucky`** | **Polar `SilW`** | **Polar `Error` (Loud)** | Caught-Exc Rate (All Runs)* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gen-0 Seed | 1 | 19.8% | 23.3% | 45.0% | 20.9% | 14.3% | 4.5% |
| `naive` | 4 | 24.1 ± 31.6% | 24.8 ± 42.2% | 12.8 ± 3.1% | 3.6 ± 3.2% | 59.5 ± 36.3% | 13.5 ± 13.4% |
| `preserve` (DarwinX) | 4 | 21.8 ± 12.9% | 18.8 ± 12.7% | 30.7 ± 21.8% | 11.8 ± 10.5% | 35.6 ± 36.1% | 19.9 ± 22.0% |
| **`metamorphic` (Ours)** | 4 | 3.2 ± 2.7% | 1.5 ± 2.0% | **10.4 ± 8.1%** | **2.7 ± 2.3%** | **83.7 ± 12.1%** | 33.0 ± 27.6% |

Key empirical findings from Polar OOD (directly answering **Risk R1** and motivating **GSB v2 / Phase P1**):
1. **Loud failure instead of silent corruption**: Because mid-latitude metamorphic invariance (`58.3°N / -56.2°S`) eliminated hardcoded zones and converged all 4 `metamorphic` seeds to dynamic `estimate_utm_crs()`, they fail **loudly** (`Error = 83.7%`, with best harness `v3_gis_s2_meta` at `99.6% Error / 0.0% Lucky / 0.4% SilW`) outside UTM's `[-80°, 84°]` domain, rather than emitting silent distorted answers like `Gen-0 Seed` (`65.9%` silent `Lucky+SilW`) or locked-in `preserve` (`42.5%` silent `Lucky+SilW`, `s2_preserve=67.7%`, `s3_preserve=72.1%`).
2. **Geodetic evaluation pitfall (why mid-latitude + polar must be evaluated jointly)**: Paradoxically, the *worst* harness on `eval480` (`v3_gis_s0_naive`, `32.9%` ALL / `50.8%` lucky) scored `71.1%` on Polar OOD! Its workspace inspector printed `utm_crs=...` on equatorial `low`, causing the LLM to hardcode an equatorial UTM zone (`EPSG:326xx`) — which fails catastrophically at mid-latitudes, yet stays within `<0.3%` distortion near the poles because every Transverse Mercator cylinder passes through both poles ($x \le R\cos 86.5^\circ \approx 389\text{ km}$).
3. **Hard boundary of fixed mid-latitude metamorphic operators**: None of the 12 v0.2 harnesses discovered local AEQD/Stereographic projections for polar latitudes, directly proving that fixed human-written mid-latitude perturbations cannot teach polar coverage — motivating **GSB v2 (polar + antimeridian)** and **Phase P1's Self-Discovering Metamorphic Verifier**.
4. \*Telemetry note (**Risk R4**): The v0.2 `eval_ood` batch logged `swallowed_count` across all tasks (including `Error` tasks where `solution.py` had a `try/except` block yet still crashed loudly, e.g., `v3_gis_s2_meta` had `99.6% Error` with 41% `caught_exceptions`); `eval_ood.py` has now been upgraded to also emit `swallowed_runnable_count` to isolate silent exception swallowing in runnable scripts.

---

## 2.1 Verified v0.3 Results: Zero-Leakage Phase Diagram, Verifier Co-Evolution & Two-Timescale 26B RSI (2026-11-21)

All 49 v4/v5 evaluation JSONs (`runs/v4_*`, `runs/v5_two_timescale/`) passed strict auditing with **`usage.errors == 0`** on full unseen `eval480` ($n = 480$) and Polar OOD Protocol A/B.

### 1. BF16 Jitter Floor & Non-Evolution Baselines (`runs/v4_jitter_check.json`, `runs/v4_block1_evals/`)
- **BF16 evaluation jitter**: Re-evaluating identical `seed.py` on `dev-36` (`gemma-4-E4B-it`, `temp=0.0`, `workers=16`) yields a `k=1` per-task flip rate of **22.2%** (`8.0/36` flips, mean `4.0` one-way regressions; `maj@3` reduces flip rate only to `16.7%`). Strict `max_regress=0` therefore rejects valid candidates even under an Oracle gate, motivating soft bounded regression ($\delta = 0.14$) for fair DarwinX comparisons.
- **Non-evolution baselines (frozen E4B no-think, `n = 480`)**: 15-minute expert `handcrafted.py` scores **74.2% ALL / 87.5% HO / 1.5% Lucky** (`1.51` calls); inference-time `runtime_meta.py` scores **70.0% ALL / 83.3% HO / 6.0% Lucky** (`2.12` calls).
- **Polar OOD Protocol B (live deployment)**: Deploying fixed mid-latitude `v3_s2_meta` directly on polar workspaces induces panic fallback to `EPSG:3857` during repair turns (`17.7%` correct, `55.0%` silent `Lucky+SilW`, `SwRun=283`).

### 2. Zero-Leakage Single-Factor Ablation & True Weak-Proxy Phase Boundary (`runs/v4_wave1_evals/`)
- **Controlled 3-seed ablation (`±7°`, `--minimal-meta-prompt`, `parent_policy=best`, $\delta=0.14$, Gemini-4 Meta → 4B Task, `n=480`)**:
  - `naive`: `75.2 ± 0.4%` (`74.8 / 75.6 / 75.2`, Lucky `5.9%`)
  - `darwinx_soft` ($\delta=0.14$): `76.9 ± 1.1%` (`78.1 / 75.8 / 76.9`, Lucky `5.9%`)
  - `meta_gate_only` (zero metamorphic diagnostic text): **`80.8 ± 2.0%`** (`81.7 / 78.5 / 82.3`, HO `93.6%`, Lucky `4.1%`) — outperforms `darwinx_soft` on 3/3 seeds (`+3.9 pp`) and `handcrafted` (`+6.6 pp`).
  - `meta_neutral` (anonymized `probe_1/probe_2`): **`84.0% / 84.6%`** (HO `98.3% / 100.0%`, Lucky `1.5% / 0.4%`) on converged seeds `s0/s1`, `55.6%` on mid-transition `s2` (4-gen cap; mean `74.7%`, lowest Lucky `2.2%`).
  - `oracle` (`max_regress=0`): `69.0 ± 15.9%` (`51.2 / 73.8 / 82.1`), confirming `max_regress=0` suffers under BF16 jitter.
- **True weak proxy (`|lat| <= 2.2°`, `EPSG:3857` area error $< 0.81\%$)**: On `v4_eq2_s1`, `darwinx_soft` ($\delta=0.14$) locked in at Gen 1 (`eval480 = 35.0% ALL / 27.5% HO / 32.5% Lucky`) while rejecting genuine Gen-2 (`OraDev=69.4%, Lucky=0.0%`) and Gen-3 (`OraDev=61.1%`) fixes; `meta_neutral` reached **`79.4% ALL / 91.7% HO / 2.3% Lucky`** (`+44.4 pp`).

### 3. Self-Certified Verifier Co-Evolution Breaking Polar OOD (`runs/v4_coev_polar/`)
- Starting from mid-latitude `v4_clean_s1_meta_neutral` (`eval480 = 84.6%`, Polar-A `0.2%`, Polar-B `13.6%`), `meta_rsi/verifier_agent.py` autonomously proposed a polar AEQD relocation (`86.5°N, 15.0°E`) and passed non-circular conservation certification on Attempt 1 (`max_area_err=0.00000%`, `max_len_err=0.00000%`, `max_dist_err=0.00123%`, `topo_mismatches=0`, `max_disp_km=11007.1`, `caught=8/8`).
- Controlled telemetry ablation: without `sys.monitoring` swallowed-exception telemetry (`solver_stage2`), evolution stalled at `16.7%` due to an `except RuntimeError: EPSG:3857` fallback; adding 5 lines of domain-neutral telemetry (`solver_stage3_telemetry`) resolved the fallback in 3 generations (`in_loop 11.1% → 88.9%`), yielding **`eval480 = 87.3% ALL / 98.3% HO / 0.6% Lucky / 1.7% Err` (`1.16` calls)**, **Polar-A `85.9%` (`+85.7 pp`)**, and **Polar-B `82.1%` (`+68.5 pp`, `SilW=0.7%`)**.

### 4. Weight-Level GRPO Pilot on E4B (`runs/v4_grpo/`, 60 steps, 1,920 rollouts, bare `seed.py`)
- `GRPO-Weak` nearly doubled in-training weak reward (`0.259 → 0.498`) by burning `EPSG:3857` into weights (`20.0% → 42.2%`, peak `84.4%`), inflating `eval480` `Lucky` (`31.2% → 38.3%`, HO `42.5%`) while degrading HO accuracy (`27.5% → 24.2%`) and yielding `57.9%` `SilW` on Polar OOD-B.
- `GRPO-Metamorphic` blocked `389/1,920` shortcuts, held `eval480` `Lucky` to `29.8%` (`-8.5 pp` vs `GRPO-Weak`) and cut Polar OOD-B `SilW` by `-12.2 pp` (`45.7%`), but exposed a sparse-reward exploration bottleneck from scratch (`218/1,920 = 11.4%` full-reward rollouts, `eval480 = 27.1%`) — motivating Two-Timescale RSI.

### 5. 100% Open-Weight Symmetric 26B-A4B RSI & Dual-G4 Two-Timescale Closed Loop (`runs/v4_sym26b_evals/`, `runs/v5_two_timescale/`)
All stages below use solely open-weight `google/gemma-4-26B-A4B-it` (`no-think`), zero closed APIs, zero 3-location oracle labels during training/evolution, and strictly disjoint splits (`dev = train[0:6]`, `distill/GRPO = train[20:55]`, `test = eval480`):

| Stage | Model Weights | Eval Harness | Avg Calls | **Test ALL (`n=480`)** | **Seen (`n=360`)** | **Held-out (`n=120`)** | **`Lucky`** | **`SilW`** | **`Err`** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gen-0 Baseline | Base 26B | Bare `seed.py` | 1.00 | 36.5% | 38.9% | 29.2% | 39.2% | 19.6% | 4.8% |
| Gen-1 `darwinx_soft` (s0 / s1) | Base 26B | Evolved Harness | 1.08 / 1.01 | **20.0% / 20.4%** | 24.7% / 25.3% | **5.8% / 5.8%** | **63.7% / 59.0%** | 16.0% / 19.0% | 0.2% / 1.7% |
| Distillation Control (`SFT-Weak`) | `sft_weak` LoRA | **Bare `seed.py`** | **1.00** | **22.5%** | 25.6% | **13.3%** | **59.0%** | 17.7% | 0.8% |
| Gen-1 `meta_neutral` (s1 Teacher) | Base 26B | Evolved Harness | 1.50 | 86.2% | 86.7% | 85.0% | 7.5% | 6.2% | 0.0% |
| Internalization Stage 1 (`SFT-Meta`) | `sft_meta` LoRA | **Bare `seed.py`** | **1.00** | **88.3%** | 85.8% | **95.8%** | **1.0%** | 4.0% | 6.7% |
| Internalization Stage 2 (`+GRPO-Meta`) | `grpo_meta_final` | **Bare `seed.py`** | **1.00** | **92.1%** | 90.3% | **97.5%** | **0.4%** | 3.8% | 3.8% |
| **Gen-2 Compounding RSI** | `grpo_meta_final` | **Gen-2 Harness (zero GIS rules)** | **1.09** | **93.3%** | **91.7%** | **98.3%** | **0.4%** | 5.8% | **0.4%** |

Key takeaways:
- **Inverse scaling of Proxy Lock-In**: Under symmetric open-weight evolution, `26B-A4B` starts at `DevLow=80.6%` (`OraDev=33.3%`) and under `darwinx_soft` actively writes `EPS:3857` into its own prompt (`2/2` seeds degrade from `36.5% → 20.0% / 20.4%`); distilling those weak trajectories (`SFT-Weak`, `95.0%` `EPSG:3857`) permanently degrades bare weights to `22.5%` (`Lucky 59.0%`).
- **Complete Two-Timescale compounding**: Metamorphic trajectory distillation (`537/630` kept) eliminates the sparse-reward bottleneck (`21–32/32` full rewards per GRPO step), raises single-turn bare weights to `88.3%` (`SFT-Meta`) and `92.1%` (`+24` steps Dual-G4 online `GRPO-Metamorphic`, `42.9s/step`, `56.7 GB` VRAM), and enables a completely domain-agnostic Gen-2 error-recovery harness to reach **`93.3%` ALL / `98.3%` Held-out / `0.4%` Lucky / `0.4%` Err** at `1.09` calls (`12/21` families at `100%`), confirming GSB v1 saturation and motivating **GSB v2**.

---

## 2.2 Verified v0.4 Results: GSB v2 Calibration, Leak-Free Multi-Round Verifier Co-Evolution, Real OSM Transfer & Non-GIS `SciUnits` (2026-11-21)

All 16 evaluation JSONs (`runs/v6_gsb_v2_s3/`, `runs/v6_gsb_v2_s3b/`, `runs/v6_gsb_v2_s3b_stage2/`) passed strict auditing with **`usage.errors == 0`**.

### 1. GSB v2 Benchmark (`data/v2`, 1,950 Tasks) & 4-Tier Separation (`n = 450`: 300 Seen + 150 Held-Out)
- **Deterministic audit (`docs/gsb_v2_audit.log`)**: `reference_pass = 1,950/1,950 (100%)`, `utm_pass = 0/1,950`, `webmercator_pass = 0/1,710` on projection/raster/large families while `wm_low_pass = 95.1%` (CompV) and `93.0%` (Rast) under `|lat| <= 1.5°`, `naive_nodata_caught = 660/660 (100%)`.
- **Single-seed 4-tier ladder on `data/v2` (`google/gemma-4-26B-A4B-it` no-think vs `E4B`)**:

| Configuration | Model | **ALL (`n=450`)** | **Seen (`n=300`)** | **HO (`n=150`)** | **`Lucky`** | **`SilW`** | **`Err`** | Calls | `CompV` | `Rast` | `Large` | `Ctrl` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot `seed.py` | E4B | 11.3% | 13.7% | 6.7% | 21.3% | 38.7% | 28.7% | 1.00 | 2.8% | 5.6% | 14.3% | 67.5% |
| Static `handcrafted.py` | E4B | 19.8% | 21.7% | 16.0% | 9.8% | 33.6% | 36.9% | 1.91 | 0.0% | 10.6% | 48.6% | 85.0% |
| Zero-shot `seed.py` | **26B-A4B** | 30.7% | 33.0% | 26.0% | 44.4% | 14.2% | 10.7% | 1.00 | 2.2% | 31.9% | 65.7% | 92.5% |
| Weak Verifier `darwinx_soft` (s0) | **26B-A4B** | 32.2% | 32.3% | 32.0% | **54.0%** | 10.0% | 3.8% | 1.21 | 1.1% | 35.6% | 65.7% | 97.5% |
| Static `handcrafted.py` | **26B-A4B** | 35.1% | 40.7% | 24.0% | 11.3% | 4.0% | **49.6%** | 1.15 | 0.0% | 30.0% | 100.0% | 100.0% |
| Fixed Mid-Lat `T_0` (`stage1_meta_T0`, s0) | **26B-A4B** | 22.9% | 27.3% | 14.0% | 4.9% | 17.1% | **55.1%** | 1.17 | 1.7% | 27.5% | 22.9% | 100.0% |
| **Self-Certified Co-Ev (`stage2_coev`, s0)** | **26B-A4B** | **62.9%** | **64.3%** | **60.0%** | 12.7% | 22.7% | **1.8%** | **1.18** | **82.8%** | **54.4%** | 12.9% | 95.0% |

- **2nd-order Proxy Lock-In exposed**: Fixed mid-latitude `T_0` solved 1st-order `EPSG:3857` lock-in (`InLoop(T_0) = 73.3%`, `Lucky: 44.4% → 4.9%`), but converged to `gdf.estimate_utm_crs()` and hit the exact same polar/antimeridian wall (`Err = 55.1%`, `CompV = 1.7%`) as static `handcrafted.py` (`Err = 49.6%`).
- **Strictly Hint-Free Multi-Round Verifier Discovery**: After stripping all failure-mode keywords from `VERIFIER_SYSTEM_PROMPT` (unit-tested) and adding vector–raster co-location conservation checks:
  - `gemini-4-hydrogen` (`thinking_level=low`) autonomously discovered and certified **3 orthogonal probes** passing 15/15 GSB v2 dev families: (1) **Polar `86°N` LAEA** co-located vector+raster relocation (`caught=8/12`, `max_len_err=0.00087%`, `max_dist_err=0.0225%`), (2) **Web Mercator `EPSG:3857` projected GPKG** exposing `if crs.is_geographic:` bypass (`0.00000%` error), and (3) **Non-meter projected CRS (`+units=km`)** on both `.gpkg` and `.tif` (`0.00000%` error).
  - 100% open-weight `26B-A4B` (`enable_thinking=False`) also self-certified a cross-layer CRS-mismatch probe on Attempt 1 (`out_tok=1,578`, `0.00000%` error across 10/10 workspaces, `caught=1/12`), whereas `enable_thinking=True` without a token budget hit 8,192-token thinking truncation on 6/6 attempts.
- **Stage-2 Co-Evolution**: Adding the 3 self-certified probes immediately closed the `43.3 pp` verifier–oracle gap (`InLoop` dropped `73.3% → 26.7%`, matching `OraDev = 26.7%`) and lifted `eval450` by **`+40.0 pp`** (`22.9% → 62.9%`, `HO: 14.0% → 60.0%`, `CompV: 1.7% → 82.8%`, `Rast: 27.5% → 54.4%`, `Err: 55.1% → 1.8%`).

### 2. Zero-Shot Real OpenStreetMap 4-Latitude Urban Transfer (`data/osm_eval/`, `n = 24`)
Fetched directly from the live OpenStreetMap v0.6 API across 4 real cities spanning `1.29°N` to `69.65°N` (Web Mercator area distortion `1.007x → 1.468x → 4.028x → 8.226x`):

| Evolved Harness (Trained Only on Synthetic GSB v2) | **ALL (`n=24`)** | Singapore (`1.29°N`) | Xuzhou (`34.26°N`) | Helsinki (`60.17°N`) | Tromsø (`69.65°N`) | `SilW` | `Err` | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot `seed.py` | 41.7% | 5/6 | 3/6 | 1/6 | 1/6 | 41.7% | 16.7% | 1.00 |
| Weak Verifier `darwinx_soft` | **29.2%** | **6/6** | 1/6 | **0/6** | **0/6** | **70.8%** | 0.0% | 1.29 |
| Fixed Mid-Lat `T_0` (`stage1_T0`) | 95.8% | 5/6 | 6/6 | 6/6 | 6/6 | 4.2% | 0.0% | 1.29 |
| **Self-Certified Co-Ev (`stage2_coev`)** | **100.0%** | **6/6** | **6/6** | **6/6** | **6/6** | **0.0%** | **0.0%** | **1.17** |

### 3. Non-GIS Scientific Benchmark Calibration (`benchmarks/sci_units/`, `data/sci_v1`, 760 Tasks)
- 12 scientific/engineering tabular+`metadata.json` families (8 seen + 4 held-out) auditing physical unit conversions, ADC calibration `(scale, offset)`, missing-value sentinels (`-999/-9999`), and row/column permutations (`ref_pass = 760/760`, `naive_low_pass = 760/760`, `naive_all_pass = 0/760`).
- Zero-shot `26B-A4B` `seed.py` (`n=280` unseen test): **`ALL = 7.1%` (`HO = 0.0%`, `Lucky = 45.4%`, `Err = 18.2%`)**.
- Weak verifier `darwinx_soft` (5 gens): `DevLow/InLoop` climbed monotonically `68.8% → 87.5%` (`Dev Lucky: 56.2% → 75.0%`) while `OraDev` remained frozen at `12.5%` and unseen test stalled at **`ALL = 7.1%` (`HO = 0.0%`, `Lucky = 44.6%`, `SilW = 47.9%`)** — reproducing Proxy Lock-In identically outside GIS.

---

## 2.3 Verified v0.5 Results: Sprint 4 6-Seed GSB v2 Matrix, Controls & Verifier-Upgrade Hysteresis (2026-11-21)

All 34 Sprint 4 Wave 1A & 1B evaluation JSONs (`runs/v7_s4_wave1a_g4{a,b}/`, `runs/v7_s4_wave1b_g4{a,b}/`) passed strict auditing with **`usage.errors == 0`** (`python -m meta_rsi.eval.sprint4_summary`).

### 1. GSB v2 Multi-Seed Comparison & Single-Seed Controls (`eval450`, symmetric `26B-A4B → 26B-A4B`)

| Method / Gate | Seeds | **ALL (`n=450`)** | **Seen (`n=300`)** | **Held-Out (`n=150`)** | **`Lucky`** | **`Err`** | **`CompV`** | **`Rast`** | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot `seed.py` | 1 | 30.7% | 33.0% | 26.0% | 44.4% | 10.7% | 2.2% | 31.9% | 1.00 |
| Static `handcrafted.py` | 1 | 35.1% | 40.7% | 24.0% | 11.3% | 49.6% | 0.0% | 30.0% | 1.15 |
| Inference-time `runtime_meta.py` | 1 | 57.1% | 57.3% | 56.7% | 21.6% | 10.9% | 49.4% | 44.4% | 2.61 |
| `naive` (s0) | 1 | 46.9% | 49.7% | 41.3% | 31.6% | 8.4% | 22.8% | 40.0% | 1.43 |
| `meta_gate_only` (zero diagnostic text, s0) | 1 | 53.6% | 50.0% | 60.7% | 13.6% | 16.7% | 55.6% | 44.4% | 1.42 |
| `oracle_soft` (3-location oracle upper bound, s0) | 1 | 55.3% | 53.7% | 58.7% | 22.4% | 12.0% | 46.1% | 40.6% | 1.44 |
| `darwinx_soft` (weak `dev_low`) | 6 | 31.9 ± 1.2% | 33.7 ± 2.1% | 28.3 ± 4.4% | 50.3 ± 3.6% | 5.6 ± 3.7% | 1.3 ± 1.1% | 37.7 ± 3.2% | 1.11 ± 0.09 |
| `meta_T0` (fixed mid-latitude probe) | 6 | 23.6 ± 5.2% | 28.8 ± 2.6% | 13.0 ± 10.6% | 18.4 ± 22.1% | 41.2 ± 25.3% | 0.7 ± 0.7% | 29.8 ± 4.9% | 1.22 ± 0.23 |
| `llm_judge` (static code-review rubric) | 6 | 39.5 ± 14.4% | 40.9 ± 11.6% | 36.7 ± 21.4% | 19.8 ± 10.0% | 17.6 ± 18.8% | 20.8 ± 21.2% | 37.1 ± 7.7% | 1.32 ± 0.10 |
| `coev_tip_only` (Stage-2 from Stage-1 tip alone) | 6 | 37.1 ± 17.7% | 40.8 ± 14.4% | 29.6 ± 25.0% | 14.4 ± 10.8% | 27.4 ± 26.4% | 28.9 ± 30.7% | 36.1 ± 11.6% | 1.31 ± 0.18 |

Key empirical observations from Wave 1A/1B:
- **6/6 Weak-Verifier Lock-In (`darwinx_soft`)**: Across all 6 paired seeds (`32.2 / 31.1 / 31.8 / 32.0 / 34.0 / 30.4%`), soft regression gating on `dev_low` remains strictly locked in around `31.9 ± 1.2%` (`CompV = 1.3 ± 1.1%`, `Lucky = 50.3 ± 3.6%`).
- **6/6 Second-Order Lock-In under Fixed Mid-Lat Verifier (`meta_T0`)**: All 6 seeds fall at or below `seed.py` (`22.9 / 17.6 / 30.0 / 19.1 / 29.6 / 22.2%`, mean `23.6 ± 5.2%`); every converged run (`s0, s1, s3, s5`) adopts `gdf.estimate_utm_crs()` and crashes on polar/antimeridian test tasks (`Err = 55–60%`, `CompV = 0.0–1.7%`).
- **Static LLM-as-a-Judge (`llm_judge`) Variance & UTM Blindspot**: Static rubric evaluation produces high seed variance (`30.0 / 51.1 / 48.0 / 32.7 / 56.2 / 19.1%`, mean `39.5 ± 14.4%`). In `s5`, the judge approved a harness prescribing `gdf.estimate_utm_crs()` (`InLoop = 63.3% vs OraDev = 26.7%`), causing the same second-order polar/antimeridian collapse (`ALL = 19.1%`, `Err = 54.2%`).
- **Zero-Label Counterfactual Guidance vs. Oracle & Inference-Time Baselines**: At seed 0, `stage2_coev` (`62.9%`, `1.18` calls) outperforms both the 3-location `oracle_soft` upper bound (`55.3%`, `1.44` calls) and inference-time `runtime_meta.py` (`57.1%` at `2.61` calls, 2.2x inference cost), while zero-diagnostic `meta_gate_only` (`53.6%`, `HO = 60.7%`) nearly matches `oracle_soft`.
- **Verifier-Upgrade Hysteresis (`coev_tip_only` vs. Archive-Preserving Co-Evolution)**: Restarting Stage 2 strictly from the Stage-1 UTM-locked tip (discarding `gen_000`) broke out on `s0` (`62.9%`), `s2` (`44.0%`), and `s5` (`46.2%`), but remained trapped at `g00` on `s1` (`16.7%`) and `s3` (`18.7%`). Audit of Wave 1B logs also identified two domain-neutral infrastructure issues — (i) `_strip_thinking_preamble` splitting on unquoted `</think>`/`<channel|>` inside Python comments (discarding 20–50% of proposals with `NameError`), and (ii) `build_failure_digest()` omitting `workspace_files` and `task_prompt` (blinding the Meta-Agent to `metadata.json` on `SciUnits`, where all 3 seeds of both `darwinx_soft` and `meta_soft` stayed at `7.1%`). Both fixes plus archive-preserving re-scoring (`extra_seed_harnesses` retaining `gen_000` alongside Stage-1 ancestors) are verified in unit tests and queued for Wave 1C.

---

## 3. Open Methodological Risks & Mitigation Tracker

| ID | Risk / Reviewer Critique | Current Status | Concrete Mitigation |
|---|---|---|---|
| **R1** | **Target-location overlap**: `metamorphic.py` relocates local inputs to `(58.3°, 6.0°)` / `(-56.2°, 12.0°)`, overlapping GSB v1 `high`/`zone_edge`. | **Resolved via Polar OOD Protocol A/B & Self-Certified Co-Evolution (2026-11-21); GSB v2 next** | Demonstrated self-certified polar verifier discovery breaking Polar-A (`0.2% → 85.9%`) and Polar-B (`13.6% → 82.1%`) with zero polar hints or labels; expanding to polar + antimeridian + raster/OSM in GSB v2. |
| **R2** | **Diagnostic hint leakage in v0.2**: `evaluator.py` previously appended a GIS/CRS hint string inside `meta_diag`. | **Resolved (2026-11-21)** | Verified `--minimal-meta-prompt` + anonymized `probe_1/probe_2` AND zero-diagnostic `meta_gate_only` (`80.8%`, 3/3 seeds > `darwinx_soft` `76.9%`). |
| **R3** | **Missing baselines & seed count**: Need non-evolution baselines, Oracle/soft-DarwinX/LLM-judge controls, and ≥6–8 seeds. | **Non-evolution, soft-DarwinX & Oracle completed (2026-11-21); `llm_judge` & ≥6 seeds planned on GSB v2** | Completed `handcrafted` (`74.2%`), `runtime_meta` (`70.0%`), `darwinx_soft` ($\delta=0.14$), and `oracle`; scale full multi-seed matrix + `llm_judge` on GSB v2. |
| **R4** | **Swallowed-exception degradation**: Rule-heavy prompts and OOD fallbacks silently swallow exceptions. | **Resolved & ablated (2026-11-21)** | Domain-neutral `sys.monitoring` telemetry broke 2nd-order `except RuntimeError: EPSG:3857` lock-in during polar co-evolution (`16.7% → 88.9%` in-loop). |
| **R5** | **Bounded iteration vs true RSI**: Fixed human-written `metamorphic.py` + closed-API Meta-Agent. | **Core mechanisms verified (2026-11-21)** | Verified (a) non-circular self-certified verifier co-evolution, (b) 100% open-weight symmetric `26B → 26B` evolution, and (c) Dual-G4 Two-Timescale Harness → SFT + GRPO-Metamorphic → Gen-2 compounding (`36.5% → 93.3%`). |

---

## 4. Phased Technical Roadmap

### Phase P0 — Immediate Methodological Rigor & Controls
- [x] **Complete v0.2 Evaluation Matrix** *(done 2026-11-20)*: Full 480-task test eval + 440-task polar OOD across Gen-0 and 12 `v3_gis` harnesses.
- [x] **Zero-Leakage Ablation, BF16 Jitter & Non-Evolution Baselines (R2/R3)** *(done 2026-11-21)*: Quantified `k=1` 22.2% BF16 flip floor; completed `handcrafted`, `runtime_meta`, `darwinx_soft` ($\delta=0.14$), `meta_gate_only`, `meta_neutral`, `oracle`, and `±2.2°` true weak-proxy runs.
- [x] **Literature Review Package**: Verified `DarwinX`, `Hyperagents`, `When Self-Evolution Backfires`, `Double Ratchet`, `Verifier Hardening`, `GISAgentBench`, `GeoAnalystBench`, `GeoContra`; documented in [LITERATURE.md](LITERATURE.md).
- [x] **GSB v2 & Real OSM Validation Slice (A1)** *(done 2026-11-21)*: Built & audited 1,950-task GSB v2 (polar, antimeridian, `|lat| <= 1.5°` weak proxy, raster/DEM, composite vector pipelines) and live OpenStreetMap 4-latitude urban slice (`data/osm_eval/`, 24 tasks across Singapore, Xuzhou, Helsinki, Tromsø).

### Phase P1 — True RSI Depth (Beyond Bounded Harness Iteration)
- [x] **Self-Certified Metamorphic Verifier Co-Evolution (v1 Polar + GSB v2 Multi-Round)** *(done 2026-11-21)*: Verified strictly hint-free zero-label certification of 3 orthogonal GSB v2 probes + open-weight `26B-A4B` self-certified probe, closing the 43.3 pp verifier–oracle gap (`22.9% → 62.9%` on `eval450`, `100.0%` on real OSM).
- [x] **Two-Timescale RSI & Dual-G4 Split 26B Closed Loop (v1 Proof — A3)** *(done 2026-11-21)*: Built reusable `meta_rsi/training/` (`lora_sync.py`, `collect_trajectories.py`, `split_two_timescale.py`), verified 100% open-weight `26B → 26B` compounding (`36.5% → 86.2% → 88.3% [SFT] → 92.1% [+GRPO] → 93.3% [Gen-2]`, vs `SFT-Weak` collapse to `22.5%`).
- [ ] **Two-Timescale Weight Internalization on GSB v2 (Wave S4-2)**: Run Dual-G4 Split `SFT-Weak` vs `SFT-Meta` + `GRPO-Metamorphic` (with certified probes) on GSB v2 `train[20:60]` and evaluate bare `seed.py` on `eval450`, real OSM, and `GeoAnalystBench`.

### Phase P2 — External Generalization & Publication
- [x] **6-Seed GSB v2 Matrix + `llm_judge` & `oracle_soft` Controls** *(Wave 1A/1B done; Wave 1C Archive-Preserving Co-Ev & fixed `SciUnits` queued)*: Completed 6 seeds each of `darwinx_soft`, `meta_T0`, `llm_judge`, `coev_tip_only`, plus `runtime_meta`, `naive`, `meta_gate_only`, `oracle_soft`.
- [ ] **External Benchmark Transfer**: Run deterministic CRS/metric/raster audit on 17 open-source `GeoAnalystBench` Python tasks (`meta_rsi/eval/geoanalyst_eval.py`) across Base, `SFT-Weak`, `SFT-Meta`, and `GRPO-Metamorphic` 26B weights.
- [ ] **Non-GIS Cross-Domain Generalization (`SciUnits`)**: Run 3-seed fair `darwinx_soft` vs `meta_soft` (with `workspace_files` + `prompt` digest) and zero-label tabular verifier co-evolution on `data/sci_v1`.

---

## 5. Local Run & Artifact Registry (Gitignored Data)

- `data/v1/`: 2,700 train + 480 test tasks (`tasks.jsonl` + variant directories).
- `results/zero_shot_v1/`: 5 zero-shot baselines (`e4b_think`, `e4b_think_hint`, `e4b_think_rules`, `q27_low`, `g26b_think`), rule & `gemini-decision-1` error labels, `caught_{base,rules}.json`.
- `runs/v3_gis_s{0..3}_{naive,preserve,meta}/`, `runs/eval480/`, `runs/eval_ood/`: Canonical v0.2 4-seed archive and full-480 / Polar-440 evals.
- `runs/v4_jitter_check.json`, `runs/v4_split_2g4_bench.json`: BF16 jitter floor and Dual-G4 split feasibility measurements.
- `runs/v4_block1_evals/`, `runs/v4_wave1_evals/`, `runs/v4_coev_polar/`, `runs/v4_sym26b_evals/`, `runs/v4_sym26b_harnesses/`, `runs/v4_grpo/`: Complete v4 ablation, co-evolution, symmetric 26B, and E4B GRPO artifacts (all 45 JSONs `usage.errors == 0`).
- `runs/v5_two_timescale/`: Complete A3 Dual-G4 26B Two-Timescale RSI artifacts — disjoint `train[20:55]` trajectories, `sft_weak` / `sft_meta` / `grpo_meta_final` summaries & full `eval480` JSONs, Gen-2 archive & `gen2_best_harness.py`, and trained bf16 LoRA checkpoint (`lora_grpo_meta_final/`, all 4 eval JSONs `usage.errors == 0`).

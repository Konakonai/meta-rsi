# `meta-rsi` — Long-Term Research Plan, Experiment Log & Roadmap

> Living master document for the `meta-rsi` research project, undergraduate thesis, and SCI / top-venue manuscript.
> Repository: `https://github.com/Konakonai/meta-rsi`
> Last updated: 2026-11-27.
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
- **Deterministic audit (`docs/gsb_v2_audit.log`)**: `reference_pass = 1,950/1,950 (100%)`, `utm_pass = 0/1,950`, `webmercator_pass = 0/1,710` on projection/raster/large families while `wm_low_pass = 95.1%` (CompV) and `93.0%` (Rast) under `|lat| <= 1.5°`, `naive_nodata_caught = 627/660 (95.0%)` (33 naive scripts still pass: `raster_pop_exposure` 27, `raster_zonal_mean` 5, `raster_corridor_elevation` 1; 0 errors).
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

## 2.3 Verified v0.5–v0.6 Results: Sprint 4 Full 6-Seed GSB v2 Matrix, Archive-Preserving Co-Evolution, 6-Seed OSM Transfer, `SciUnits` & `GeoAnalystBench` Baseline (2026-11-21 – 2026-11-25)

All 59 Sprint 4 Wave 1A, 1B, and 1C evaluation JSONs (`runs/v7_s4_wave1{a,b,c}_g4{a,b}/`) passed strict auditing with **`usage.errors == 0`** and **`usage.calls >= n`** (`python -m meta_rsi.eval.sprint4_summary`; regenerated tables and figures in `runs/figures/`). Wave 1C ran cleanly on Vertex AI serverless MaaS (`google/gemma-4-26b-a4b-it-maas`, `api_key="gcloud"`, `VERTEX_MIN_INTERVAL_S=3.35`, parity-verified `24/24` on real OSM against Colab G4 vLLM), preserving `637.74 CU` on Colab for Wave S4-2 Dual-G4 weight training.

### 1. GSB v2 Multi-Seed Comparison & Single-Seed Controls (`eval450`, symmetric `26B-A4B → 26B-A4B`)

> **Superseded (2026-11-26).** The unified one-server re-evaluation in §2.6.1 replaces this table as the main 6-seed result. The numbers here are kept as originally reported and reproduce within 0.8 pp per method. In the unified analysis, `arch_coev − naive` is +6.7 pp (sign-flip p = 0.25, not established), and every comparison between single methods below is descriptive. Seed 0 was a favourable seed for the evolved harnesses themselves (§2.6.1), so the seed-0 comparisons are one seed out of six.

| Method / Gate | Seeds | **ALL (`n=450`)** | **Seen (`n=300`)** | **Held-Out (`n=150`)** | **`Lucky`** | **`Err`** | **`CompV`** | **`Rast`** | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot `seed.py` | 1 | 30.7% | 33.0% | 26.0% | 44.4% | 10.7% | 2.2% | 31.9% | 1.00 |
| Static `handcrafted.py` | 1 | 35.1% | 40.7% | 24.0% | 11.3% | 49.6% | 0.0% | 30.0% | 1.15 |
| Inference-time `runtime_meta.py` | 1 | 57.1% | 57.3% | 56.7% | 21.6% | 10.9% | 49.4% | 44.4% | 2.61 |
| `naive` (s0; 6 seeds: 34.4 ± 6.2%, §2.5) | 1 | 46.9% | 49.7% | 41.3% | 31.6% | 8.4% | 22.8% | 40.0% | 1.43 |
| `meta_gate_only` (zero diagnostic text, s0; 6 seeds: 37.2 ± 8.4%, §2.5) | 1 | 53.6% | 50.0% | 60.7% | 13.6% | 16.7% | 55.6% | 44.4% | 1.42 |
| `oracle_soft` (3-location oracle upper bound, s0; 6 seeds: 39.3 ± 10.7%, §2.5) | 1 | 55.3% | 53.7% | 58.7% | 22.4% | 12.0% | 46.1% | 40.6% | 1.44 |
| `darwinx_soft` (weak `dev_low`) | 6 | 31.9 ± 1.2% | 33.7 ± 2.1% | 28.3 ± 4.4% | 50.3 ± 3.6% | 5.6 ± 3.7% | 1.3 ± 1.1% | 37.7 ± 3.2% | 1.11 ± 0.09 |
| `meta_T0` (fixed mid-latitude probe) | 6 | 23.6 ± 5.2% | 28.8 ± 2.6% | 13.0 ± 10.6% | 18.4 ± 22.1% | 41.2 ± 25.3% | 0.7 ± 0.7% | 29.8 ± 4.9% | 1.22 ± 0.23 |
| `llm_judge` (static code-review rubric) | 6 | 39.5 ± 14.4% | 40.9 ± 11.6% | 36.7 ± 21.4% | 19.8 ± 10.0% | 17.6 ± 18.8% | 20.8 ± 21.2% | 37.1 ± 7.7% | 1.32 ± 0.10 |
| `coev_tip_only` (Stage-2 from Stage-1 tip alone) | 6 | 37.1 ± 17.7% | 40.8 ± 14.4% | 29.6 ± 25.0% | 14.4 ± 10.8% | 27.4 ± 26.4% | 28.9 ± 30.7% | 36.1 ± 11.6% | 1.31 ± 0.18 |
| **`arch_coev` (Archive-Preserving Co-Ev, Ours)** | **6** | **41.0 ± 13.6%** | **44.0 ± 12.0%** | **35.0 ± 18.0%** | **19.1 ± 14.5%** | 25.4 ± 20.7% | **21.1 ± 19.0%** | 37.1 ± 9.9% | 1.39 ± 0.27 |

Key empirical observations from Wave 1A/1B/1C:
- **6/6 Weak-Verifier Lock-In (`darwinx_soft`)**: Across all 6 paired seeds (`32.2 / 31.1 / 31.8 / 32.0 / 34.0 / 30.4%`), soft regression gating on `dev_low` remains strictly locked in around `31.9 ± 1.2%` (`CompV = 1.3 ± 1.1%`, `Lucky = 50.3 ± 3.6%`).
- **6/6 Second-Order Lock-In under Fixed Mid-Lat Verifier (`meta_T0`)**: All 6 seeds fall at or below `seed.py` (`22.9 / 17.6 / 30.0 / 19.1 / 29.6 / 22.2%`, mean `23.6 ± 5.2%`); every converged run (`s0, s1, s3, s5`) adopts `gdf.estimate_utm_crs()` and crashes on polar/antimeridian test tasks (`Err = 55–60%`, `CompV = 0.0–1.7%`).
- **Static LLM-as-a-Judge (`llm_judge`) Variance & UTM Blindspot**: Static rubric evaluation produces high seed variance (`30.0 / 51.1 / 48.0 / 32.7 / 56.2 / 19.1%`, mean `39.5 ± 14.4%`). In `s5`, the judge approved a harness prescribing `gdf.estimate_utm_crs()` (`InLoop = 63.3% vs OraDev = 26.7%`), causing the same second-order polar/antimeridian collapse (`ALL = 19.1%`, `Err = 54.2%`).
- **Zero-Label Counterfactual Guidance vs. Oracle & Inference-Time Baselines**: At seed 0, `stage2_coev` (`62.9%`, `1.18` calls) outperforms both the 3-location `oracle_soft` upper bound (`55.3%`, `1.44` calls) and inference-time `runtime_meta.py` (`57.1%` at `2.61` calls, 2.2x inference cost), while zero-diagnostic `meta_gate_only` (`53.6%`, `HO = 60.7%`) nearly matches `oracle_soft`. *Correction (§2.5): seed 0 was the best seed for `naive`, `meta_gate_only` and `oracle_soft`; over 6 seeds no method beats `naive` (`arch_coev` +6.7 pp, sign-flip p = 0.22).*
- **Archive-Preserving Co-Evolution (`arch_coev`) Resolves Verifier-Upgrade Hysteresis**: Retaining `gen_000` alongside re-scored Stage-1 ancestors (`extra_seed_harnesses` with `parent_gen=None` and fixed DGM fallback pool in `meta_rsi/archive.py`) lifted breakout rate from **3/6** (`coev_tip_only`: `62.9 / 16.7 / 44.0 / 18.7 / 34.0 / 46.2%`, SD `17.7%`) to **5/6** (`arch_coev`: `58.2 / 17.1 / 37.8 / 45.6 / 45.6 / 41.8%`, SD `13.6%`), rescuing previously trapped seeds `s3` (`18.7% → 45.6%`, `+26.9 pp`) and `s4` (`34.0% → 45.6%`, `+11.6 pp`). Across all 6 seeds, `arch_coev` leads all multi-seed methods (`41.0% ALL`, `44.0% Seen`), outperforms fixed `meta_T0` (**`+17.4 ± 12.8 pp` ALL**, paired $t=3.33, p=0.0207$, 95% CI `[+8.1, +26.5]`; exact sign-flip p = 0.0625, since seed 1 differs by −0.5 pp; **`+22.0 ± 14.6 pp` HO**, $t=3.70, p=0.0140$, 95% CI `[+10.9, +31.9]`), and improves over `darwinx_soft` by `+9.1 pp` while slashing `Lucky` by `-31.2 pp` (`50.3% → 19.1%`) and lifting `CompV` **16.2×** (`1.3% → 21.1%`). On the single remaining trapped seed (`s1`, `17.1%`), `g01` recombined `estimate_utm_crs()` from `g-1` and `g02..g08` misdiagnosed the polar `RuntimeError` as an empty filtered GeoDataFrame.

### 2. Zero-Shot Real OpenStreetMap 4-City Urban Transfer (6 Seeds, `n = 24`)
Evaluating all 6 Wave 1C `arch_coev` harnesses zero-shot on live OpenStreetMap slices across Singapore (`1.29°N`), Xuzhou (`34.26°N`), Helsinki (`60.17°N`), and Tromsø (`69.65°N`):
- **`arch_coev` (6 seeds)**: **`86.1 ± 26.2%` ALL**, **`0.0 ± 0.0%` Error**, **`13.9 ± 26.2%` SilW**, `1.28 ± 0.41` calls (`s0=91.7%, s1=100.0%, s2=33.3%, s3=100.0%, s4=91.7%, s5=100.0%`) — **5/6 seeds achieve `91.7%–100.0%`** (vs. `seed.py` `41.7%` and weak `darwinx_soft` `29.2%` with `70.8%` silent wrong).

### 3. Non-GIS Cross-Domain Generalization (`SciUnits`, 3 Paired Seeds, `eval280`)

| Configuration | Seeds | **ALL (`n=280`)** | **Seen (8)** | **Held-Out (4)** | **`Lucky`** | **`SilW`** | **`Err`** | Calls | Per-Seed ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `darwinx_soft` (`26B → 26B`, fixed digest) | 3 | 7.1 ± 0.0% | 12.5 ± 0.0% | 0.0 ± 0.0% | 57.5 ± 6.5% | 34.6 ± 5.6% | 0.7 ± 0.9% | 1.15 ± 0.01 | `7.1 / 7.1 / 7.1%` |
| `meta_soft` (`26B → 26B`, fixed digest) | 3 | 7.1 ± 0.0% | 12.5 ± 0.0% | 0.0 ± 0.0% | 59.3 ± 2.9% | 33.6 ± 2.9% | 0.0 ± 0.0% | 1.16 ± 0.05 | `7.1 / 7.1 / 7.1%` |
| `darwinx_soft` (`gem4h → 26B`, weak SI dev) | 3 | 7.1 ± 0.0% | 12.5 ± 0.0% | 0.0 ± 0.0% | **91.4 ± 0.4%** | 0.5 ± 0.5% | 1.0 ± 0.8% | 1.35 ± 0.05 | `7.1 / 7.1 / 7.1%` |
| **`meta_soft` (`gem4h → 26B`, unit/sentinel probe)** | **3** | **10.4 ± 7.5%** | **14.4 ± 9.0%** | **5.0 ± 7.3%** | **75.4 ± 15.3%** | 10.8 ± 18.8% | 3.5 ± 3.2% | 1.20 ± 0.02 | **`9.6 / 3.2 / 18.2%`** |

Key non-GIS mechanistic findings (zero domain hints, `minimal_meta_prompt=True`):
1. **Meta-Agent synthesis bottleneck under symmetric `26B → 26B`**: Even with the fixed `workspace_files` + `task_prompt` digest, `26B` diagnosed unit/sentinel drift in rationales but never synthesized runtime `metadata.json` file-inspection code in 6 generations (`7.1%` across all 3 seeds for both gates).
2. **Pure Goodhart lock-in under `darwinx_soft` (`gem4h → 26B`)**: With a capable domain-neutral Meta-Agent (`gemini-4-hydrogen`), `darwinx_soft` drove `dev_low` to `100.0%` on 3/3 seeds and inflated unseen `Lucky` to **`91.4 ± 0.4%`** (Held-Out `Lucky = 96.7%–97.5%`), scoring **`0.0%` on all 11 physics families** (`7.1%` = `control_weighted_sensor_mean` only).
3. **Autonomous non-GIS discovery & isomorphic 2nd-order verifier-coverage gap (`meta_soft`, `gem4h → 26B`)**: Under label-free tabular invariance gating, `meta_soft` autonomously invented `inspect_workspace()`, dynamic `metadata.json` loading, `missing_sentinel` sanitization, affine `calibration`, and `SI_FACTORS` lookup (`in_loop`: `12.5% → 50.0%` on `s0`, `12.5% → 75.0%` on `s2`), slashing `SilW` to **`0.0%`** on both `s0` and `s2`, and breaking out on `s2` to **`18.2%` ALL / `13.3%` Held-Out / `0.0%` Err** (`pipe_darcy_headloss 45.0%`, held-out `projectile_range_envelope 46.7%`, `beam_bending_stress 15.0%`, `hydrostatic_dam_force 10.0%`). Code audit of `s0`/`s2` reveals *why* `Lucky` remained high (`86.4%` / `81.8%`): `_TABULAR_UNIT_PAIRS` in `meta_rsi/metamorphic.py` applies a single deterministic mostly-metric pair per SI unit (`Pa → kPa`, `m3 → L`, `K → degC`, `W → kW`), leaving Imperial test units (`psi`, `ft3`, `gal_us`, `degF`, `hp`, `lbf`, `lb/ft3`, `in4`) uncovered — reproducing the exact same static-verifier coverage boundary as GSB v2 `meta_T0` in a completely non-GIS scientific domain.

### 4. Wave S4-2: GSB v2 Two-Timescale Weight Internalization, Gen-2 Compounding RSI, Real OSM & External `GeoAnalystBench` Transfer (`runs/v8_s4_wave2_v2/`, 2026-11-25)

All 12 Wave S4-2 evaluation JSONs (open-weight `google/gemma-4-26B-A4B-it` BF16 for every trained and evaluated model; the 3 certified probes used to filter trajectories and in the GRPO reward were certified by the closed `gemini-4-hydrogen` on Colab G4 96 GB, strictly disjoint `train[20:60]` = 600 tasks vs dev `[0:12]` and unseen test `n=450`, zero 3-location oracle labels during training) passed strict auditing with **`usage.errors == 0`** and **`usage.calls >= n`**:
- **Disjoint trajectory collection (`train[20:60]`, 600 tasks)**: `weak_trajectories.jsonl` (`v2_sym26b_s0_darwinx_soft_best.py`, weak `dev_low` filter) kept `493/600` (`48.7%` / `240` scripts using `EPSG:3857` shortcuts); `meta_trajectories.jsonl` (`stage2_coev_best.py` + 3 self-certified probes) blocked `58` subtle shortcuts and kept `391/600` (`0.26%` / `1` `EPSG:3857`).
- **Online `GRPO-Metamorphic` (4 alternating rounds × 6 steps = `24` steps × `32` rollouts = `768` rollouts, exact Schulman k3 KL vs frozen `sft_ref`, 3 certified probes)**: `438/768` (`57.0%`) full metamorphic passes, `65` shortcuts blocked, **`0/768` (`0.0%`) `EPSG:3857`**, `final KL = 0.00369`.

| Stage | Model Weights | Eval Harness | Calls | **GSB v2 ALL (`n=450`)** | **Seen (`n=300`)** | **HO (`n=150`)** | **`Lucky`** | **`Err`** | **`CompV`** | **`Rast`** | **`Large`** | **Real OSM (`n=24`)** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gen-0 Baseline | Base 26B | Bare `seed.py` | 1.00 | 30.7% | 33.0% | 26.0% | 44.4% | 10.7% | 2.2% | 31.9% | 65.7% | 41.7% (10/24) |
| Weak Control (`SFT-Weak`) | `sft_weak` LoRA | **Bare `seed.py`** | **1.00** | **30.4%** | 33.0% | **25.3%** | **48.0%** | 11.6% | **1.1%** | 32.5% | 62.9% | **12.5%** (3/24; 0/12 polar/high) |
| Gen-1 Teacher | Base 26B | `stage2_coev` | 1.18 | 62.9% | 64.3% | 60.0% | 12.7% | 1.8% | 82.8% | 54.4% | 12.9% | 100.0% (24/24) |
| Stage 1 (`SFT-Meta`) | `sft_meta` LoRA | **Bare `seed.py`** | **1.00** | **56.2%** | 61.3% | **46.0%** | **10.7%** | 14.7% | **69.4%** | **45.0%** | 22.9% | **79.2%** (19/24; 12/12 Hel/Tro) |
| Stage 2 (`+GRPO-Meta`, 24 steps) | `grpo_meta_final` | **Bare `seed.py`** | **1.00** | **62.4%** | **63.7%** | **60.0%** | **11.3%** | **7.6%** | **83.3%** | **46.2%** | **24.3%** | **91.7%** (22/24; 0.0% Err) |
| **Gen-2 Compounding RSI** | **`grpo_meta_final`** | **Generic Retry (0 GIS rules)** | **1.09** | **`68.2%`** | **`67.7%`** | **`69.3%`** | 12.4% | **`0.7%`** | **`88.3%`** | **`55.0%`** | **28.6%** | **`100.0%`** (24/24; 1.04 calls) |

- **External Zero-Shot Transfer on `GeoAnalystBench` (`n = 17` metric/CRS/raster Python tasks, fair `max_tokens = 4096`, `100.0% syntax_ok` across all 4 weights, `uerr = 0`)**:

| Weights (Zero-Shot External Prompt) | `syntax_ok` | `uses_3857` | `proj_metric_vec` | **`dynamic_crs_vec`** | **`raster_nodata_safe`** | `crs_aligned` | **`strict_geo_sound`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base 26B (`base_26b`) | 100.0% | 5.9% | 85.7% | **0.0%** | 75.0% | 100.0% | 76.5% |
| **`SFT-Weak` LoRA (`sft_weak_26b`)** | 100.0% | 0.0% | 85.7% | **0.0%** | **37.5%** (`-37.5 pp`) | 100.0% | **64.7%** (`-11.8 pp`) |
| **`SFT-Meta` LoRA (`sft_meta_26b`)** | 100.0% | **0.0%** | 85.7% | **`71.4%`** (`+71.4 pp`) | **`87.5%`** (`+12.5 pp`) | 100.0% | **`88.2%`** (`+11.7 pp` vs Base, **`+23.5 pp` vs SFT-Weak**) |
| **`GRPO-Meta` LoRA (`grpo_meta_26b`)** | 100.0% | **0.0%** | 85.7% | **`71.4%`** (`+71.4 pp`) | 75.0% | 100.0% | **`82.3%`** (`+5.8 pp` vs Base, **`+17.6 pp` vs SFT-Weak**) |

Key Wave S4-2 breakthroughs:
1. **Complete GSB v2 Two-Timescale Compounding**: `GRPO-Metamorphic` lifts single-turn bare `seed.py` from `30.7% → 56.2% (SFT-Meta) → 62.4% (+GRPO-Meta, +31.7 pp ALL, +34.0 pp HO, CompV 83.3% > Teacher 82.8%)` while `SFT-Weak` stays locked at `30.4%` (`Lucky = 48.0%`). Adding a completely domain-agnostic Gen-2 retry harness (zero GIS rules!) pushes **`grpo_meta_final` to `68.2%` ALL / `69.3%` Held-Out / `0.7%` Err** (`+5.3 pp` ALL and **`+9.3 pp` Held-Out above the multi-turn Gen-1 Teacher `62.9% / 60.0%`**!).
2. **`+79.2 pp` Real-World OpenStreetMap Weight Separation**: With zero GIS prompt rules (`1.00` call), `SFT-Weak` collapses real 4-city OSM from `41.7% → 12.5%` (`0/6` Helsinki, `0/6` Tromsø, `70.8%` silent wrong), whereas `SFT-Meta` reaches `79.2%`, `GRPO-Meta` reaches **`91.7%` (22/24, `0.0%` Err)**, and Gen-2 Compounding hits **`100.0%` (24/24, `1.04` calls)**!
3. **True Out-of-Distribution External Transfer (`GeoAnalystBench`)**: On completely unseen open-source `GeoAnalystBench` workflows (never seen in any training or evolution split), metamorphic weight internalization drives `dynamic_crs_vec` from **`0.0% → 71.4%`** (`+71.4 pp`), eliminates `EPSG:3857` (`5.9% → 0.0%`), and lifts `strict_geo_sound` from `76.5% → 88.2%` (`SFT-Meta`) / `82.3%` (`GRPO-Meta`) while `SFT-Weak` degrades Base 26B to `64.7%` (`-11.8 pp`, halving `raster_nodata_safe` `75.0% → 37.5%`).

*Correction (§2.5):* the `SFT-Weak` vs `SFT-Meta` gap above is a teacher effect. Trained on the `stage2_coev` teacher's scripts with the weak filter, `sft_same_teacher_weak` matches `sft_meta` (Protocol A 60.2 vs 58.2%, Protocol B 50.2 vs 50.2%, 8-city OSM 96.9 vs 92.7%). A weak-reward GRPO run from the same init matches `GRPO-Meta`, and 5 SFT epochs beat it on Protocol A.

## 2.4 Metamorphic Gate False-Rejection Audit (`runs/v11_gate_precision/`, 2026-11-25)

How often does the gate behind `base_meta`, `sft_meta` and `meta_gate_only` (built-in `shift_and_shuffle` and `crs_storage_flip` probes, then the 3 certified probes) reject a correct solution? Two CPU-only measurements, both re-running the exact `_run_metamorphic_check` used in collection and evolution (`scripts/rebuttal/gate_precision.py`, `scripts/rebuttal/gate_reference_osm.py`).

### 1. Gate vs. oracle on unfiltered trajectories (GSB v2 `train[20:60]`, 1,384 scripts)

Every script passed only the weak check (correct on `low`). Its gate verdict is compared with the oracle (correct on every hidden variant). The re-run reproduces the collection-time filter: all 35 `base_meta` scripts whose code is identical to a `base_weak` script pass, and 143 of the 149 `base_weak` scripts the gate passes belong to tasks that `base_meta` kept.

| Trajectory source | Oracle-correct | False rejection, full gate | False rejection, built-in probes only | False acceptance, full gate | False acceptance, built-in only | Oracle-correct among passed |
|---|---|---|---|---|---|---|
| Bare 26B `seed.py` (`base_weak`) | 183/444 | 19.7% (36/183) | 7.1% (13/183) | 0.8% (2/261) | 3.8% (10/261) | 98.7% |
| `stage2_coev` teacher (`same_teacher_weak`) | 400/447 | 5.8% (23/400) | 0.2% (1/400) | 8.5% (4/47) | 36.2% (17/47) | 99.0% |
| `darwinx_soft` s0 teacher (Wave S4-2 `SFT-Weak` data) | 181/493 | 20.4% (37/181) | 6.6% (12/181) | 1.3% (4/312) | 4.2% (13/312) | 97.3% |

- The certified probes trade false rejections for false acceptances: they cut false acceptance (36.2% → 8.5% for the strong teacher) and raise false rejection about 3× for the two weaker sources (7.1% → 19.7%, 6.6% → 20.4%) and from 0.2% to 5.8% for the strong teacher.

**Why the gate rejects oracle-correct scripts** (`scripts/rebuttal/gate_rejection_causes.py`). Each rejection is replayed in the workspace of the probe that rejected it. The family's reference solver is run on that workspace too, which shows whether the probe preserves the answer. Each rejection gets the first matching cause:

| Cause | Bare 26B | `stage2_coev` teacher | `darwinx_soft` teacher |
|---|---|---|---|
| Gate error: probe changes the reference answer | 2 | 2 | 2 |
| Gate error: polar probe writes a polygon that crosses the antimeridian unsplit (violates RFC 7946 §3.1.9) | 4 | 0 | 4 |
| Gate error: both answers within 1% of their own truths, but 1.01% apart (gate tolerance) | 0 | 0 | 1 |
| Layers left in different CRSs and combined without reprojecting (all `topological_containment_v2`) | 8 | 0 | 9 |
| Projected coordinates read as metres (km probe; consistent again when the same probe uses metres) | 0 | 17 | 0 |
| Stored coordinates treated as WGS84 degrees, or measured in the stored projection (Web Mercator / transverse Mercator storage) | 13 | 2 | 14 |
| Continental scene moved over the pole: every polygon valid, but the layer spans more than 180° of longitude | 2 | 0 | 4 |
| Other: answer preserved, script drifts 1–13% or crashes (not traced) | 7 | 2 | 3 |
| Total false rejections | 36 | 23 | 37 |

- 15 of the 96 rejections (16%) are gate errors. *Corrected after independent review (`runs/agent_mail/carbon_review_1.md`); the first version counted 6.* The three kinds:
  - **Resampled thresholds (6).** The same two `raster_pop_exposure` tasks in every source: the polar probe resamples the hazard raster, some blocks cross the threshold, and the reference answer changes with them; the scripts follow it exactly.
  - **Malformed probe output (8).** The polar probe relocates large regions over the pole and writes polygons whose rings run from about −177° to +177° without being split at the antimeridian, which RFC 7946 §3.1.9 requires. A standards-following script reads such a ring as spanning ~350° of longitude (e.g. a global equal-area projection, `EPSG:6933`, returns 5.3M km² for a 0.54M km² region). The reference survives only because `pyproj.Geod` takes the short way along each edge. Per-polygon check by re-running the probe: 6 of the 11 polar-wrap tasks contain such a polygon.
  - **Gate tolerance (1).** The gate requires the two answers to agree within 1%, while the oracle accepts each within 1% of its own truth, so two oracle-correct answers can be up to ~2% apart and still be rejected.
- The review counted 23–29 gate errors. The additional cases it named are scripts whose probe-workspace answer misses the probe's reference answer, so they stay script failures under the rule used here. One example is `raster_pop_exposure-train-0035` under `shift_and_shuffle`: 26,583 against a reference 23,534. The review's point that the probe stores that raster in Web Mercator, against the family's own high-latitude convention, is fair, but the reference handles the stored data.
- Most rejections are scripts that fail on an input whose answer the probe preserves, in a regime the oracle variants never contain:
  - layers stored in different CRSs, which these tasks' variants never do (they store every layer in one CRS);
  - km-unit storage;
  - large-extent data stored in a projection, e.g. `pyproj.Geod` applied to raw projected x/y, which returns NaN;
  - large regions near the pole.
- The oracle calls these scripts correct because GSB v2 never tests those inputs, not because the scripts handle them. "False rejection" in the table above therefore means disagreement with the benchmark's oracle; against a stricter oracle most of these would be true rejections.
- Correction to the first reading of these results: the `large_*` truths differ across *oracle variants* (the shapes are translated in degrees), but the probes that rejected `large_*` scripts change the reference answer by at most 0.005%. Those rejections fall under the storage and polar rows, not under non-invariance.
- The remaining polar row (6 rejections) is the least clear-cut. Every polygon is well formed, but the scene sits over the pole and spans more than 180° of longitude, a case the oracle's large-extent variants (±140° longitude) exclude. The polygons have 7 sparse vertices, with edges spanning 50° of longitude or more. Reprojecting only the vertices to a planar or equal-area CRS cuts off the poleward bulge of those geodesic edges, which puts the scripts' areas a few percent off. Geodesic ring-area methods handle it, and so do equal-area projections after the edges are densified along the geodesic.
- Fixes for the gate: split relocated geometries at the antimeridian (or reject a probe that emits a single non-point geometry wider than 180°), and widen the gate's agreement tolerance to about twice the oracle's.
- Effect on distillation data: the bare model writes 25 oracle-correct `large_region_area_v2` scripts, and the metamorphic filter keeps 2 of them in `base_meta`, so `sft_base_meta` sees almost no examples of that family that the oracle grades correct. In this audit, 21 of the bare model's 36 rejections are `large_region_area_v2` scripts, and most of them fail under projected storage or over the pole (rows above).

### 2. Reference solutions on real OpenStreetMap tasks (`data/osm_global` + `data/osm_eval`, 120 tasks)

One reference solution per template (local azimuthal-equidistant projection; reproduces all 120 stored truths), with every probe run and scored separately:

| Template | Gate pass | Rejecting probes |
|---|---|---|
| 9 metric templates (area, length, perimeter, buffers, nearest neighbour, density) | 96/96 | — |
| `osm_northernmost_bldg_lat` (absolute position) | 0/8 | `shift_and_shuffle`, `crs_storage_flip`, `probe_r2_polar_laea`: the answer becomes the probe's target latitude (58.3, −56.2, 86.0) |
| `osm_north_half_bldg_count` (absolute position) | 7/8 | `probe_r2_polar_laea` (1) |
| `osm_bldg_road_intersects_cnt` (exact-topology control) | 3/8 | `crs_storage_flip` (5): the reprojection round trip breaks shared-vertex contacts and counts drop by 2–6 |

Implications: the gate is a high-precision filter (97–99% of passed scripts are oracle-correct). On GSB v2, 15 of its 96 rejections are its own errors: a probe that changes the answer, emits malformed geometry, or applies a tolerance tighter than the oracle's. Most of what it rejects fails on inputs the oracle does not test, so the measured false-rejection rate mostly reflects the oracle's coverage. The assumption that every probe preserves the answer does fail for answers tied to absolute position (the real-OSM latitude tasks) and for probes that are not lossless: thresholded aggregates over a resampled raster, and exact topology after a reprojection round trip. Results built on the gate should report its false-rejection rate against the oracle together with this breakdown.

## 2.5 Rebuttal Controls for Weight Internalization and the Completed 6-Seed Baselines (`runs/v9_rebuttal_gpu/`, 2026-11-25)

All runs: `google/gemma-4-26B-A4B-it` bf16 with LoRA on Colab G4 vLLM, temperature 0, no thinking. All 52 eval JSONs have `usage.errors == 0`, `calls >= n` and per-task records. Training uses the same disjoint `train[20:60]` (600 tasks) and SFT recipe (2 epochs) as Wave S4-2 unless noted. The analysis below pairs models task by task, with 95% bootstrap intervals that resample task families.

### 1. Controls

| Model | Init | Training data | Size |
|---|---|---|---|
| `sft_same_teacher_weak` | base | `stage2_coev` teacher (the `sft_meta` teacher), weak filter (correct on `low`) | 447 (1 uses EPSG:3857) |
| `sft_base_weak` | base | bare 26B `seed.py`, weak filter | 444 (191 use EPSG:3857) |
| `sft_base_meta` | base | bare 26B `seed.py`, metamorphic filter | 147 (3 use EPSG:3857) |
| `sft_meta_more` | base | the S4-2 `sft_meta` data (teacher + metamorphic filter), 5 epochs instead of 2 | 391 |
| `grpo_weak_final` | `sft_meta` | GRPO on the `grpo_meta_final` schedule (4 rounds × 6 steps × 32 rollouts), reward = correct on `low` only | 768 rollouts (487 rewarded, 1 uses EPSG:3857) |

The S4-2 models (base, `sft_weak`, `sft_meta`, `grpo_meta_final`) were evaluated again in the same session. Base, `sft_meta` and `grpo_meta_final` have 3 Protocol A repeats.

### 2. Results

Protocol A: one script per task, written on `low` and scored on every variant. Protocol B: the model writes a new script for each variant, and the task counts only if all of them are right.

| Model | GSB v2 Protocol A | Protocol B | B held-out | B `Lucky` | OSM, 8 cities (`n=96`) | OSM, 4 cities (`n=24`) |
|---|---:|---:|---:|---:|---:|---:|
| Base 26B | 29.0% (27.8 / 30.0 / 29.1) | 27.3% | 25.3% | 47.6% | 54.2% | 41.7%* |
| `sft_weak` (`darwinx_soft` teacher) | 30.4%* | 26.7% | 21.3% | 45.8% | 36.5% | 12.5%* |
| `sft_base_weak` | 31.6% | 28.9% | 23.3% | 50.7% | 51.0% | 29.2% |
| `sft_base_meta` | 32.4% | 30.4% | 29.3% | 48.2% | 67.7% | 54.2% |
| `sft_meta` | 58.2% (58.7 / 59.3 / 56.7) | 50.2% | 36.7% | 14.9% | 92.7% | 79.2%* |
| `sft_same_teacher_weak` | 60.2% | 50.2% | 44.7% | 15.6% | **96.9%** | **95.8%** |
| `sft_meta_more` | **66.9%** | **59.3%** | **64.7%** | **8.0%** | 87.5% | 70.8% |
| `grpo_meta_final` | 62.3% (61.6 / 64.4 / 60.9) | 56.4% | 50.0% | 12.0% | 90.6% | 91.7%* |
| `grpo_weak_final` | 63.3% | 52.4% | 43.3% | 16.2% | 91.7% | **95.8%** |

\* Earlier evaluations without per-task records (Wave S4-2; base OSM from `runs/v6_gsb_v2_s3b_stage2/`), not paired below.

| Comparison (paired by task) | What it isolates | Protocol A | Protocol B | OSM, 8 cities |
|---|---|---:|---:|---:|
| `sft_same_teacher_weak` − `sft_meta` | filter, same teacher | +2.0 [−4.7, +8.6] | +0.0 [−7.2, +7.6] | +4.2 [+0.0, +10.4] |
| `sft_meta` − `sft_weak` | teacher (and filter) | — | +23.6 [+6.9, +40.5] | +56.2 [+34.4, +77.1] |
| `sft_base_meta` − `sft_base_weak` | filter, bare-model data | +0.9 [−4.7, +6.5] | +1.6 [−2.7, +6.6] | **+16.7 [+2.1, +33.3]** |
| `sft_base_meta` − base | self-distillation | +3.5 [−0.1, +7.9] | +3.1 [−0.4, +7.5] | **+13.5 [+3.1, +25.0]** |
| `grpo_weak_final` − `grpo_meta_final` | GRPO reward | +1.0 [−5.5, +7.3] | −4.0 [−10.4, +0.9] | +1.0 [−12.5, +15.6] |
| `grpo_meta_final` − `sft_meta` | GRPO over its init | +4.1 [−0.2, +9.0] | +6.2 [+0.2, +12.6] | −2.1 [−16.7, +11.5] |
| `sft_meta_more` − `sft_meta` | 5 vs 2 SFT epochs | +8.7 [+2.6, +14.9] | +9.1 [+0.7, +18.0] | −5.2 [−18.8, +3.1] |
| `sft_meta_more` − `grpo_meta_final` | longer SFT vs GRPO | +4.6 [+0.7, +9.3] | +2.9 [−2.7, +8.3] | −3.1 [−22.9, +14.6] |

Findings:
- **The S4-2 gap between `SFT-Weak` and `SFT-Meta` comes from the teacher, not the filter.** With the same `stage2_coev` teacher, the weak filter does as well as the metamorphic one under both protocols and on both OSM sets. The two teachers differ in quality: 37% of the `darwinx_soft` teacher's kept scripts are oracle-correct (181/493, §2.4) against 89% for `stage2_coev` (400/447). The strong teacher writes almost no EPSG:3857 scripts (1/447 kept by the weak filter), so the metamorphic filter has little to remove. The gate's contribution is limited to producing the teacher harness under it, and even that rests on one seed. `stage2_coev` is seed 0 of `coev_tip_only` (62.9%; the other five seeds score 16.7–46.2%, and over 6 seeds the method does not beat `naive`, §2.5.3). The weak control's teacher, `darwinx_soft` s0, scores 32.2%. These results measure distillation from the best single harness, not an across-seed gain. Remaining differences: `sft_meta` was trained in the S4-2 session on 391 samples, `sft_same_teacher_weak` here on 447.
- **On bare-model data the filter mostly shrinks the training set to easy families.** The bare model rarely solves the multi-step families in one try, so 142 of the 147 `base_meta` scripts come from 4 families (`district_mean_zero_trap_v2` 40, `raster_zonal_mean` 40, `raster_pop_exposure` 36, `topological_containment_v2` 26). Nine families contribute none. One sample per task cannot show whether the filter helps where the model can write candidates.
- **On bare-model data the metamorphic filter helps on real OSM only.** On GSB v2 it is within noise (+0.9 / +1.6). On the 8-city OSM set, `sft_base_meta` beats `sft_base_weak` by 16.7 pp using a third of the data (147 vs 444 scripts), with the gains in building area (38 → 100%), large-building area (25 → 100%), road length and hull density (25 → 62%). This is the only significant effect of the filter itself in these controls.
- **The GRPO reward does not matter from this init.** Weak reward and metamorphic reward from the same `sft_meta` end within noise on Protocol A and OSM. Protocol B leans toward the metamorphic reward (−4.0, interval includes 0). Rollouts from `sft_meta` almost never use EPSG:3857 (1/768 under the weak reward), so there is little for the metamorphic reward to block.
- **More SFT epochs match or beat GRPO.** Five epochs on the same 391 samples beat `grpo_meta_final` on Protocol A (+4.6) and give the best Protocol B result (59.3%, held-out 64.7%). The GRPO gain over its init (+4.1 A, +6.2 B) is no larger than what longer SFT gives. On OSM neither helps; `sft_meta_more` drops on `osm_road_buffer_poi_count` (88 → 12%).
- **Protocol B is 6–11 pp below Protocol A for every distilled model** (intervals exclude 0) and 1.6–2.7 pp below for the base-like models. Per-variant accuracy is nearly the same under both protocols (e.g. `sft_meta` high 74 / 72%, polar 69 / 68%). The gap comes from requiring every independently generated script to be right: the distilled models' errors vary from one generation to the next, while the base-like models make the same mistake on every variant (high / polar / antimeridian 26–37% in both protocols).
- **Repeat noise at temperature 0.** Between any two of the 3 Protocol A repeats, 4.9% (base), 13.6% (`sft_meta`) and 11.7% (`grpo_meta_final`) of tasks change outcome on average. Across all three repeats, 7.3%, 20.4% and 17.6% of tasks change at least once, and ALL moves by up to 3.6 pp. Single-run differences of a few points are within this.
- **Regressions shared by every model distilled from the strong teacher** (`sft_meta`, `sft_same_teacher_weak`, `sft_meta_more`, both GRPO models): `large_region_area_v2` and `large_route_length_v2` are at 0% under both protocols (base 20% / 10% in A), and `large_max_geodesic_span` drops from 97% to 53–87% (A). Self-distillation goes the other way on `large_region_area_v2`: `sft_base_weak` 95%, `sft_base_meta` 55% (A). The 40-pp cost of the metamorphic filter there matches §2.4, where it kept 2 of the bare model's 25 oracle-correct scripts for that family. The cause of the strong-teacher regression is in the data: the `stage2_coev` teacher's kept trajectories contain 0 of 40 `large_region_area_v2` and 0 of 40 `large_route_length_v2` tasks, under both the weak and the metamorphic filter. Three raster families (`raster_threshold_area`, `raster_slope_exceed_area`, `raster_hotspot_in_buffer`) stay at 0–7% for every model.
- **8-city OSM** (Helsinki, Longyearbyen, Quito, Reykjavík, Singapore, Sydney, Tromsø, Wellington; 12 templates). Strong-teacher models reach 87.5–96.9% against 54.2% for base. `sft_weak` falls below base (−17.7 [−32.3, −4.2]): distilling the weak teacher hurts transfer. With the absolute tolerance (`a79afd1`), the latitude template scores 100% for every model. `osm_nearest_poi_median_dist` is the hardest template (0–88%).

### 3. Completed 6-seed baselines (`runs/v9_rebuttal_gpu/missing_6seed/`)

Seeds 1–5 of `naive`, `meta_gate_only` and `oracle_soft`, with the same 6-attempt search budget as seed 0, evaluated on local vLLM. Together with §2.3.1 every method now has 6 seeds:

| Method | ALL (`n=450`) | Seen | Held-out | `Lucky` | `Err` | `CompV` | `Rast` | Calls | Per-seed ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `naive` | 34.4 ± 6.2% | 35.8 ± 6.8% | 31.4 ± 5.2% | 47.8 ± 8.7% | 3.2 ± 3.7% | 4.4 ± 9.0% | 37.6 ± 2.0% | 1.13 ± 0.15 | 46.9 / 33.3 / 31.1 / 31.3 / 31.6 / 32.0 |
| `meta_gate_only` | 37.2 ± 8.4% | 37.9 ± 6.4% | 35.7 ± 12.8% | 39.9 ± 13.4% | 7.3 ± 6.3% | 12.3 ± 21.3% | 38.2 ± 3.8% | 1.37 ± 0.39 | 53.6 / 29.6 / 37.3 / 33.3 / 33.8 / 35.6 |
| `oracle_soft` | 39.3 ± 10.7% | 41.3 ± 7.7% | 35.2 ± 18.2% | 33.9 ± 10.7% | 7.5 ± 2.9% | 20.1 ± 21.3% | 37.2 ± 4.2% | 1.17 ± 0.15 | 55.3 / 42.0 / 26.9 / 36.2 / 45.8 / 29.3 |
| `arch_coev` (§2.3.1) | 41.0 ± 13.6% | 44.0 ± 12.0% | 35.0 ± 18.0% | 19.1 ± 14.5% | 25.4 ± 20.7% | 21.1 ± 19.0% | 37.1 ± 9.9% | 1.39 ± 0.27 | 58.2 / 17.1 / 37.8 / 45.6 / 45.6 / 41.8 |

- **Seed 0 was the best seed** for these three methods (46.9 / 53.6 / 55.3% against seed 1–5 means of 31.9 / 33.9 / 36.0%), and it is also high for `coev_tip_only` and `arch_coev`. The seed-0 comparisons in §2.3.1 (`stage2_coev` 62.9% > `oracle_soft` 55.3% > `meta_gate_only` 53.6% > `naive` 46.9%) were made on that seed.
- **No method beats `naive` over 6 seeds.** Seed-paired exact sign-flip test (the smallest possible p with 6 seeds is 0.031): `arch_coev` +6.7 pp (5/6 seeds, p = 0.22), `oracle_soft` +4.9 (p = 0.16), `meta_gate_only` +2.8 (5/6, p = 0.16), `llm_judge` +5.2 (p = 0.44), `coev_tip_only` +2.7 (p = 0.63). Only `meta_T0` differs (−10.8, 0/6, p = 0.031). `arch_coev` is +1.8 over `oracle_soft` (p = 0.88) and +3.8 over `meta_gate_only` (p = 0.38).
- On seeds 1–5, `naive` stays at 31–33%, the level of `darwinx_soft` and bare `seed.py`.
- **Unequal search budgets.** `arch_coev` and `coev_tip_only` run 8 Stage-2 generations on top of a 6-generation Stage-1 archive, 14 candidate generations in all; every other method runs 6. On seeds 0, 3 and 5, `arch_coev`'s final harness was accepted in generation 7 or 8. The 3 certified probes used by `meta_gate_only`, `coev_tip_only` and `arch_coev` were certified once by `gemini-4-hydrogen` and reused for every seed.
- Earlier confound, now resolved: `arch_coev` was evaluated on Vertex MaaS and the other methods on local vLLM, and the seed-0 evals carried no per-task records. The unified re-evaluation (§2.6) reran all 48 harnesses on one server. Every method moved by at most 0.8 pp, and the conclusion above is unchanged: `arch_coev` +6.7 pp, sign-flip p = 0.25.


## 2.6 Unified 6-Seed Re-Evaluation, Blind Relocation and Held-Out Mechanisms (`runs/v10_unified_6seed/`, 2026-11-26)

**Setup.**
- All 48 best harnesses (8 methods × 6 seeds) were re-run on one server: Colab G4 vLLM bf16, `google/gemma-4-26B-A4B-it`, temperature 0, no thinking.
- Protocol A ran twice on all 450 test tasks. Protocol B ran once on the first 5 test tasks of each family (100 tasks).
- All 144 evals are clean (`usage.errors == 0`, full per-task records).
- Blind (`data/v2_blind`, 3,553 scored variants) and held-out-mechanism (`data/v2_heldout_mech`, 1,705 scored variants) scores come from running each harness's saved Protocol A scripts offline. That ran on Colab CPU hosts with identical library versions (`docs/BLIND_PROTOCOL.md`, Addendum 4).

**Pre-registration.** The analysis was fixed before any comparison was computed, in `docs/BLIND_PROTOCOL.md` (Addenda 1–4) and `docs/NEXT_EXPERIMENTS.md` (E2, Amendments 1–3). No summary was run on partial data: the three summaries ran once, in the registered order, after all 144 evals and 192 sidecars were in. Commands:
- `python -m meta_rsi.eval.unified_summary runs/v10_unified_6seed --ref naive` (and `--ref arch_coev`)
- `python scripts/rebuttal/score_blind.py summary` (and `--blind data/v2_heldout_mech summary`)

Outputs are `runs/v10_unified_6seed/{unified_summary_ref_*,blind_summary,heldout_summary}.{md,json}`.

**Decision rule (Addendum 2).**
- There is one confirmatory comparison per analysis:
  - unified: `arch_coev − naive` on Protocol A;
  - E2: `arch_coev − naive` on the far mechanism.
- Each is established only if the exact sign-flip test over 6 seeds gives p < 0.05, which needs all 6 seeds to agree.
- Everything else is descriptive. It is reported with t₅ intervals on the seed-level differences, and Bonferroni intervals over 14 comparisons where given. No descriptive comparison is called significant.

### 1. Main result: Protocol A, a task counts only if one script is right on every variant

| Method | Search generations | Calls / task | `correct` | `lucky` | `silent_wrong` | `error` | − `naive` (pp) [t₅ 95%] | t₅ Bonferroni/14 | sign-flip p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| `naive` | 6 | 1.12 | 33.8 ± 5.6% | 48.5% | 14.6% | 3.0% | — | — | — | — |
| `darwinx_soft` | 6 | 1.11 | 31.2 ± 1.4% | 51.9% | 12.0% | 4.9% | −2.6 [−8.5, +3.2] | [−14.4, +9.1] | 0.31 | 1.00 |
| `meta_T0` | 6 | 1.22 | 23.3 ± 5.4% | 18.1% | 17.3% | 41.3% | −10.5 [−19.5, −1.5] | [−28.6, +7.6] | 0.062 | 0.44 |
| `meta_gate_only` | 6 | 1.38 | 36.6 ± 7.7% | 40.5% | 15.1% | 7.7% | +2.8 [−0.7, +6.3] | [−4.3, +9.9] | 0.12 | 0.75 |
| `coev_tip_only` | 14 | 1.31 | 37.5 ± 18.1% | 14.1% | 21.2% | 27.1% | +3.7 [−11.8, +19.2] | [−27.4, +34.8] | 0.53 | 1.00 |
| `oracle_soft` | 6 | 1.19 | 39.4 ± 10.7% | 34.1% | 19.0% | 7.5% | +5.6 [−2.2, +13.4] | [−10.1, +21.4] | 0.16 | 0.78 |
| `llm_judge` | 6 | 1.30 | 39.9 ± 15.0% | 20.9% | 21.7% | 17.4% | +6.1 [−12.1, +24.3] | [−30.4, +42.6] | 0.38 | 1.00 |
| **`arch_coev` (confirmatory)** | 14 | 1.40 | 40.5 ± 13.3% | 19.6% | 14.5% | 25.4% | **+6.7 [−5.0, +18.3]** | [−16.8, +30.1] | **0.25** | 1.00 |

Mean ± sd over 6 seeds, each seed averaged over its 2 repeats. `lucky` = right on some but not all variants, `silent_wrong` = wrong on all, `error` = crashed or no code. Holm p is adjusted across the 7 methods compared with `naive`; with 6 seeds no Holm p can fall below 0.05. Search generations count candidate generations: `arch_coev` and `coev_tip_only` run 8 Stage-2 generations on a 6-generation Stage-1 archive. The 3 probes used by `meta_gate_only`, `coev_tip_only` and `arch_coev` were certified once, by `gemini-4-hydrogen`, and reused for every seed.

- **The confirmatory comparison is not established.**
  - `arch_coev − naive` = +6.7 pp. Per seed: +13.1 / −15.1 / +6.1 / +13.4 / +13.9 / +8.6. Sign-flip p = 0.25.
  - Seed 1 is the trapped seed of §2.3.1, which recombined `estimate_utm_crs()`.
  - Under Protocol B the difference is +8.7 pp [−3.8, +21.1] (p = 0.094, descriptive).
- **`arch_coev` trades silent failures for loud ones.**
  - Silent failures (`lucky` + `silent_wrong`): 34.1% against 63.1% for `naive`.
  - Errors: 25.4% against 3.0%.
  - The primary endpoint, `correct`, moves by +6.7 pp.
- **Against `arch_coev`** (descriptive), the other methods differ by:
  - `meta_T0` −17.2 [−31.2, −3.2] (p = 0.062; Bonferroni [−45.2, +10.8]);
  - `darwinx_soft` −9.3 [−22.5, +3.9];
  - `naive` −6.7;
  - `meta_gate_only` −3.9;
  - `coev_tip_only` −3.0;
  - `oracle_soft` −1.1;
  - `llm_judge` −0.6.

  The §2.3.1 result of `+17.4 pp, p = 0.0207` against `meta_T0` came from a paired t-test, which Addendum 2 does not use. It stays descriptive.
- **The unified eval reproduces the original evals.**
  - Every method's mean moves by at most 0.8 pp, and every per-seed value by at most 2.8 pp (`llm_judge` seed 4).
  - The Vertex-vs-local confound of §2.5.3 does not matter: `arch_coev` moves by −0.5 pp (per seed +0.0 / +0.7 / −1.0 / −0.9 / −0.0 / −1.9).
  - Backend check: the reused harnesses' rep 1 − rep 2 is +0.2 pp signed, 1.1 pp absolute, against 1.3 pp absolute for fresh single-call harnesses. Repeat 1 is kept.
  - Repeat noise: over 48 repeat pairs, mean |Δ `correct`| = 1.3 pp (max 5.6), and 7.6% of tasks change outcome.
- **Protocol B (live deployment, one script per variant) on its 100-task subset.**
  - That subset is 0.6–2.5 pp easier under Protocol A for every method (`naive` 35.1 against 33.8%, `arch_coev` 41.8 against 40.5%). That accounts for part of the A-to-B rise.
  - Tasks that Protocol A scored `correct` stay correct under B for 94.2% (`naive`) and 87.3% (`arch_coev`).
  - B recovers 19.5% of `arch_coev`'s Protocol A `error` tasks (29/149) and 9.7% of its `lucky` tasks, against 16.7% and 1.9% for `naive`.
  - `meta_T0` rises the most, from 23.3% to 30.5%, as its errors fall from 41.3% to 18.8%. This is consistent with scripts written on each variant's own workspace avoiding the mid-latitude `estimate_utm_crs()` choice made on `low`, though the eval does not show that directly.
- **Seed 0 is a genuinely favourable seed.**
  - The unified seed-0 gap (seed 0 minus the mean of seeds 1–5) is +13.6 / +18.2 / +18.4 pp for `naive` / `meta_gate_only` / `oracle_soft`. The original evals gave +15.0 / +19.6 / +19.3.
  - The mean, +16.7 pp, is above the 10 pp threshold of Addendum 2. So the seed-0 advantage lives in the evolved harnesses themselves, not in the eval server.
  - This re-evaluation cannot say whether that comes from the seed-0 dev split or from search luck, and no claim is made either way.
  - The single-seed comparisons of §2.3.1 are one seed out of six.

### 2. Blind relocation (descriptive): the same scripts on 8 positions per task drawn without reference to the probes

Share of scored variants correct (per variant, not all-or-nothing), mean over seeds and repeats:

| Method | designed | blind, all | area-weighted | 0–30° | 30–60° | 60–80° | 80–89° | large | UTM-valid zone | polar zone | − `naive`, all (pp) [t₅ 95%] | t₅ Bonferroni/14 | sign-flip p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| `naive` | 40.7 | 43.0 | 48.2 | 52.7 | 36.8 | 31.5 | 31.1 | 70.9 | 36.6 | 31.2 | — | — | — | — |
| `darwinx_soft` | 38.2 | 40.5 | 45.7 | 51.6 | 35.2 | 30.7 | 29.7 | 61.5 | 35.3 | 29.7 | −2.5 [−10.9, +5.9] | [−19.4, +14.4] | 0.62 | 0.62 |
| `meta_gate_only` | 44.5 | 47.0 | 51.7 | 55.9 | 43.2 | 38.5 | 36.5 | 66.8 | 42.8 | 35.6 | +4.0 [−1.8, +9.8] | [−7.7, +15.7] | 0.16 | 0.47 |
| `oracle_soft` | 49.3 | 51.6 | 56.1 | 60.4 | 51.8 | 45.9 | 42.6 | 60.7 | 50.4 | 41.1 | +8.6 [+0.4, +16.9] | [−8.0, +25.2] | 0.094 | 0.47 |
| `meta_T0` | 45.5 | 53.5 | 60.0 | 68.9 | 63.1 | 61.7 | 37.7 | 30.7 | 63.1 | 24.8 | +10.4 [−1.6, +22.5] | [−13.7, +34.6] | 0.12 | 0.47 |
| `llm_judge` | 54.4 | 58.3 | 63.0 | 65.5 | 62.4 | 56.2 | 46.2 | 63.5 | 60.1 | 41.3 | +15.3 [−3.8, +34.3] | [−23.0, +53.5] | 0.094 | 0.47 |
| `coev_tip_only` | 56.0 | 61.8 | 66.6 | 75.0 | 72.0 | 70.2 | 53.1 | 31.8 | 71.6 | 43.7 | +18.8 [+11.3, +26.2] | [+3.8, +33.7] | 0.031 | 0.22 |
| `arch_coev` | 59.5 | 65.4 | 70.7 | 73.5 | 70.4 | 68.4 | 49.1 | 67.3 | 69.9 | 39.0 | +22.4 [+11.7, +33.2] | [+0.8, +44.0] | 0.031 | 0.22 |

- **Reading 1, as narrowed in Addendum 2, holds for `arch_coev` and `coev_tip_only`: their gain is not tied to the probes' coordinates.**
  - On the blind variants, `arch_coev`'s advantage over `naive` is +22.4 pp, against +18.8 [+8.2, +29.3] on the designed variants with the same per-variant metric. All 6 seeds are positive (+8.9 to +33.1). The Bonferroni interval over 14 comparisons is [+0.8, +44.0].
  - The advantage stays where the probes are far away: +21.2 [+11.2, +31.2] in the ≥ 15° signed probe-distance bin, and +23.5 [+9.7, +37.3] at ≥ 15° in |latitude|.
  - These are descriptive results. The raw p = 0.031 is the smallest possible with 6 seeds; Holm-adjusted it is 0.22.
- **What the blind gain can be attributed to.**
  - It belongs to two-stage diagnostic co-evolution, not to archive preservation. `coev_tip_only` has the same 14 generations and the same 3 certified probes but no archive, and it already gains +18.8 pp. `arch_coev − coev_tip_only` is +3.7 pp (p = 0.44), and −2.1 pp at ≥ 15° from the probes.
  - It is not the gate alone. With 6 generations, the binary gate without diagnostics (`meta_gate_only`) gains +4.0 pp, while the static `llm_judge`, which uses no probes, gains +15.3 pp (`arch_coev − llm_judge` +7.2, p = 0.34).
  - Both co-evolution methods searched 14 candidate generations against 6 for every other method. Their probes were certified once by a closed model (`gemini-4-hydrogen`). There is no 14-generation `naive` or `llm_judge` control, so the search budget is not separated from the method.
  - In the UTM-valid zone, the fixed mid-latitude `meta_T0` (6 generations, no certified probes) already reaches 63.1%. That is +26.5 pp over `naive`, about 80% of `arch_coev`'s +33.3.
- **The gain stays inside the mechanisms the probes exercise.**
  - `arch_coev` gains +33.3 pp in the UTM-valid zone, where `naive`'s EPSG:3857 scripts break.
  - It gains only +7.7 pp [−2.1, +17.5] (p = 0.125) in the polar zone beyond the UTM limits, +2.3 pp in the equatorial weak-proxy zone, and −3.6 pp on the large families.
  - The 2 antimeridian-crossing blind scenes are listed, not analysed (Addendum 2): every method gets exactly 1 of the 2 right.
  - In absolute terms, `arch_coev` is right on 39% of polar-zone variants.
  - `meta_T0`'s fixed mid-latitude probe gives the same pattern, in a stronger form: 63% in the UTM-valid zone, 25% in the polar zone, and 31% on large families.
- **Why the blind gain (+22.4) is larger than the strict Protocol A gain (+6.7).** Three steps, recomputed on the same Protocol A records (the reviewer's decomposition, checked):
  1. Strict to per-variant on the same designed variants, `low` included: +6.7 → +13.6 pp. The strict metric needs one script to be right on every variant; the per-variant metric credits partial success.
  2. Dropping `low`: +13.6 → +18.8 pp. On `low` itself `arch_coev` is −1.1 pp behind `naive` (−12.8 on seed 1, −16.0 on seed 2). `low` is about 29% of the designed variants (per task), but the equatorial zone is under 9% of the blind ones.
  3. Designed non-`low` to blind positions: +18.8 → +22.4 pp. The designed non-`low` variants of local families put about 40% of their weight beyond 85° (380 of 980), where `arch_coev` gains least. The blind set puts about 69% in the UTM-valid zone and 22% in the polar zone.
  - On seed 1 the chain is −15.1 → +5.1 → +11.2 → +20.9. Its blind score benefits from the composition step (+9.7), since that seed's `estimate_utm_crs()` scripts fail mostly at the poles.
  - Under this decomposition the blind gain is still not a probe-coordinate artifact. But how large it is depends on the metric and on how many polar and equatorial scenes the evaluation includes.
- **Timeouts (Addendum 4).**
  - Over all methods, the first pass had 49 blind and 24 held-out timeouts. After the one retry, 15 and 7 remain.
  - Per method, `arch_coev` went from 34 → 7 blind and 17 → 3 held-out, `darwinx_soft` stayed at 8 and 4, and `llm_judge` went from 7 and 2 to 0. Every other method had at most 1, and none after the retry.
  - The remaining timeouts are at most 0.02% of a method's 42,636 blind runs.
  - The retry barely matters. 20 of `arch_coev`'s retried blind runs turned correct (0.05% of its runs). The reviewer reverted every retried record to a timeout, and `arch_coev − naive` moved by 0.05 pp on blind, 0.16 pp on E2 far and 0.07 pp on E2 near. No sign or interval changed.

### 3. Held-out failure mechanisms (E2): the same scripts on `low` scenes with an unseen storage convention

Share of scored variants correct:

| Method | `low` | near | `us_ft` | `albers` | `nodata_nan` | adjacent: `csv_lon360` | far: `raster_southup` | descriptive: `vector_z` / `multipart` | − `naive`, near (pp) [t₅ 95%] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `naive` | 79.8 | 56.5 | 56.7 | 72.4 | 41.5 | 66.2 | 71.8 | 80.7 / 77.0 | — |
| `darwinx_soft` | 81.7 | 59.5 | 58.3 | 75.5 | 44.5 | 72.7 | 72.5 | 82.6 / 80.0 | +3.1 [−12.7, +18.8] |
| `meta_gate_only` | 76.6 | 54.0 | 55.2 | 66.9 | 41.0 | 60.8 | 71.8 | 77.1 / 74.3 | −2.4 [−16.6, +11.8] |
| `oracle_soft` | 71.7 | 52.2 | 52.0 | 60.8 | 43.6 | 51.4 | 71.5 | 71.6 / 70.0 | −4.3 [−14.3, +5.7] |
| `meta_T0` | 74.2 | 59.7 | 65.3 | 74.3 | 37.8 | 56.1 | 69.9 | 75.2 / 75.1 | +3.2 [−5.3, +11.7] |
| `llm_judge` | 68.7 | 49.0 | 52.7 | 56.2 | 38.4 | 43.8 | 69.4 | 69.0 / 64.9 | −7.4 [−18.4, +3.5] |
| `coev_tip_only` | 73.9 | 60.3 | 64.0 | 72.5 | 43.5 | 53.5 | 74.2 | 74.6 / 72.5 | +3.9 [−2.4, +10.1] |
| `arch_coev` | 78.8 | 60.7 | 67.5 | 70.6 | 42.6 | 57.1 | 72.3 | 77.6 / 72.8 | +4.3 [−5.5, +14.1] |

- **The E2 confirmatory comparison is not established, and it is uninformative, as predicted.**
  - `arch_coev − naive` on `raster_southup` is +0.5 pp. Per seed: −19.7 / −1.6 / +0.9 / +14.7 / +0.9 / +7.8. p = 0.875, t₅ 95% [−11.6, +12.6].
  - Among scripts right on `low`, both methods keep ≥ 90% under the mechanism (98.8% and 99.8%). Rasterio reads south-up rasters transparently, so this comparison cannot separate the methods (Amendment 2).
  - The `bothlow` version is −0.6 pp.
  - The per-seed spread reflects differences in which raster tasks each method solves on `low`, not the mechanism.
- **No method shows a held-out gain overall.**
  - On near-transfer mechanisms (new forms of CRS, units and nodata), every interval against `naive` includes 0. `arch_coev`'s +4.3 pp is +3.8 on tasks both methods solve on `low` (p = 0.19). On the 160 raster tasks that have both near and far variants (`matched:near`) it is +0.8 [−12.5, +14.1].
  - Per mechanism, `arch_coev − naive` goes in opposite directions:
    - `us_ft` +10.9 [+1.5, +20.3] (6/6 seeds, p = 0.031, descriptive). The km-units probe exercises reading a CRS's linear unit, and that carries over from kilometres to US survey feet.
    - `albers` −1.8. Local Albers is an equal-area projection in metres, so a script that ignores the CRS and takes planar `.area` in native coordinates gets area tasks right. That is why `naive` (72.4%) and `darwinx_soft` (75.5%) score high there.
    - `nodata_nan` +1.1. Of `arch_coev` seed 0's 90 scripts that are right on `low`, 33 fail `nodata_nan`: 19 return NaN and 14 return 0.0. All 33 call `isnan` or `isfinite` somewhere, but not on the path that runs.
      - Example (traced by the reviewer, rechecked): the script for `raster_zonal_mean-test-0000` filters with `data[data != nodata]` whenever `nodata is not None`, and uses `np.isfinite` only in the `else` branch.
      - With NaN as the declared nodata, the first branch runs. `NaN != NaN` keeps every padded cell, and the mean becomes NaN.
      - The probes only ever swap one finite nodata sentinel for another, so they cannot exercise this branch.
  - On `csv_lon360`, every gated or judged method is below `naive`: `arch_coev` −9.1 [−34.0, +15.9], `llm_judge` −22.4.
  - Over all held-out variants, `llm_judge` is the only method whose interval excludes 0: −11.3 [−19.1, −3.4].
  - `nodata_nan` is the hardest mechanism for every method (38–45%).
  - Together with §2.6.2: across held-out storage conventions, the co-evolved harnesses establish no gain (near +4.3 pp, interval includes 0). There is descriptive transfer only where the fix a probe exercises applies directly (`km` → `us_ft`, +10.9 pp), and none on `nodata_nan` (+1.1) or `csv_lon360` (−9.1). This is consistent with M1's broader point that the probes and the designed variants share failure mechanisms.

### 4. What this changes

- **§2.3.1 and §2.5.3.** The 6-seed table in §2.3.1 and the 6-seed baselines of §2.5.3 are superseded by the table in §2.6.1. The original numbers stay where they were, and they reproduce within 0.8 pp per method. The leading position of `arch_coev` and its comparisons against single methods are descriptive.
- **The thesis claim, as the data now support it.**
  - Over 6 seeds, no method establishes a gain on the strict endpoint.
  - Descriptively, two-stage diagnostic co-evolution writes scripts that are right on more relocated positions. Per variant over `naive`, `coev_tip_only` gains +19 pp and `arch_coev` +22 pp, on 6/6 seeds, and the gain is not tied to the probes' coordinates.
  - That gain comes at 14 against 6 search generations, with 3 probes certified by a closed model. At 6 generations the binary gate alone (`meta_gate_only`) gives +4 pp and the static `llm_judge` +15 pp. Archive preservation adds little on blind positions (+3.7 pp, p = 0.44).
  - The gain stays within the failure mechanisms the probes exercise (Web Mercator distortion and the UTM range). It shrinks to +7.7 pp [−2.1, +17.5] beyond the UTM limits. Across held-out storage conventions it establishes nothing, apart from a descriptive +10.9 pp on `us_ft`. It also comes with more loud failures: 25% errors against 3%.


## 2.7 E1: Teacher-Free Best-of-K Self-Distillation (`runs/v12_e1_bok/`, 2026-11-27)

**Question.** Does the metamorphic filter improve distillation data beyond the weak check once the model can write several candidates? §2.5 could not tell. With a strong teacher there was nothing to filter, and with one bare-model sample per task the filter mostly shrank the data.

**Pre-registration.** The design, the analysis script and its tests were committed before any E1 data: `docs/NEXT_EXPERIMENTS.md` (E1, Amendments 1–3) and `scripts/rebuttal/e1_summary.py` (`bd52507`). The only later code change moved an import (`1996260`). The analysis ran once, after all 39 evals were in and audited.

**Setup.**
- **Candidate pool.** Bare `gemma-4-26B-A4B-it` with `seed.py` sampled K = 16 scripts per task on `train[20:60]` (600 tasks, temperature 0.8).
- **Gate.** The E1 gate: built-in plus the 3 certified probes, with the polar probe fixed (antimeridian split, pole cap, geodesic densification) and a 2% agreement tolerance.
- **Arms.** T is the set of tasks with at least one candidate that passes the weak check and the gate. For each task in T and replicate r, one random permutation of its candidates is drawn:
  - `bok_weak` keeps the first candidate correct on `low`;
  - `bok_meta` keeps the first that also passes the gate.
- **Training.** 6 replicates, each with its own draw seed and training seed, so 12 LoRAs (SFT recipe of `sft_meta`, 2 epochs). All replicates come from the one pool, so inference is conditional on it.
- **Evaluation.** All 12 LoRAs and the bare model ran on one Colab A100 80 GB vLLM server: bf16, temperature 0, 1 call per task, bare `seed.py`. The evals cover:
  - Protocol A on all 450 GSB v2 test tasks;
  - Protocol B on the 100-task subset;
  - the 8-city OSM set (96 tasks).

  The GPU differs from the unified re-evaluation's G4, so the numbers here are comparable with each other, not with §2.6.
- **Audit.** 39/39 evals are clean.

**Reported first (pre-registered).**
- |T| = 237 of 600 tasks, from 11 families. 194 of them (82%) come from 5 families:
  - `district_mean_zero_trap_v2` 40, `large_region_area_v2` 40, `raster_zonal_mean` 40;
  - `topological_containment_v2` 38, `raster_pop_exposure` 36.

  The multi-step families rarely have any candidate that passes both checks.
- The two arms keep different scripts on 36–44 of the 237 tasks (15–19%) per replicate.
- Oracle-correct share of the training scripts: `bok_weak` 81.0–81.4%, `bok_meta` 89.5–91.1%, so 20–24 more correct scripts per replicate.

**Data link (pre-registered attribution test).** Per task in T, the oracle-correct share of the candidates that pass the gate minus that of the candidates that pass the weak check, averaged within family: **+26.4 pp over 11 families, exact sign-flip p = 0.031**. The gate-filtered pool is cleaner.
- The gain sits in `large_route_length_v2` (+87.9), `area_weighted_pop_filter` (+58.1), `multi_ring_density` (+49.9), `buffer_exclusion_area` (+42.7), `raster_buffer_extract` (+30.8) and `nearest_service_gap` (+21.2).
- It is 0 in 4 of the 5 families that dominate T, and −0.2 pp in `large_region_area_v2`. In those families the weak check already selects correct scripts.
- **Sensitivity to the family weighting** (the statistic is pre-registered; this check was raised by the reviewer and rechecked).
  - Each of the 11 families counts equally, so the 6 sparse families (43 of the 237 tasks, 2–14 tasks each) carry 55% of the weight.
  - Weighted by task, pool precision rises from 81.2% to 90.3%, a gain of +9.0 pp. That equals the gap between the two arms' training scripts.
  - Dropping the two families with fewer than 5 tasks (`nearest_service_gap` 2, `area_weighted_pop_filter` 3) gives +23.4 pp (p = 0.125). Keeping only the 7 families with ≥ 10 tasks gives +13.2 pp (p = 0.50).
  - The direction holds under each weighting; the size and the p-value do not.

**Results.**

| Eval | `bok_meta` | `bok_weak` | Bare model | `meta − weak` (pp), per replicate | Mean [t₅ 95%] | Sign-flip p |
|---|---:|---:|---:|---|---|---:|
| **Protocol A, 450 tasks (confirmatory)** | 31.6 ± 1.3% | 31.6 ± 0.6% | 30.9% | +1.3 / +0.0 / −1.3 / +1.8 / +0.2 / −2.2 | **−0.0 [−1.6, +1.6]** | **1.00** |
| Protocol B, 100 tasks | 29.3 ± 1.8% | 30.5 ± 1.9% | 30.0% | +1.0 / −2.0 / −2.0 / −1.0 / −2.0 / −1.0 | −1.2 [−2.4, +0.1] | 0.125 |
| 8-city OSM, 96 tasks | 62.0 ± 5.1% | 56.6 ± 5.5% | 57.3% | −1.0 / +5.2 / +9.4 / +9.4 / +13.5 / −4.2 | +5.4 [−1.8, +12.5] | 0.16 |

Silent and loud failures, Protocol A (meta / weak / bare): silent 62.1 / 62.9 / 62.9%, errors 6.3 / 5.5 / 6.2%. On OSM, silent failures are 29.7 / 37.2 / 37.5% and errors 8.3 / 6.2 / 5.2%.

- **The confirmatory comparison is not established, and the pre-specified Reading 2 holds: no added value.** The difference is −0.0 pp: 3 replicates are positive, 2 negative and 1 tied. The family-cluster bootstrap gives [−2.4, +2.0]. As pre-registered, the claim that metamorphic filtering improves training data beyond the weak check is dropped for GSB v2.
- **Cleaner data did not turn into a better model.** The gate raised the oracle-correct share of the training scripts by about 9 pp, and of the candidate pool by 26 pp. Yet both arms end within 0.7 pp of the bare model under both Protocol A and Protocol B.
  - At this scale, the distillation barely moves the model.
  - The 237 tasks are dominated by 5 families where the two arms pick the same script.
  - The arms differ on 15–19% of tasks.

  E1 therefore shows that the filter adds nothing here. It cannot show whether the filter would add value with more data, or with data where the weak check and the gate disagree more often.
- **What E1 isolates (raised by the reviewer, rechecked).** Building T already uses the gate at the task level, and both arms are restricted to T.
  - The bare model has a weak-passing candidate on 553 of the 600 tasks, but those candidates are oracle-correct only 36.5% of the time (task average).
  - On the 316 tasks where some candidate passes the weak check and none passes the gate, the weak-passing candidates are 2.9% oracle-correct, and about 70% of them use EPSG:3857.
  - Restricting to T removes these shortcut-only tasks from `bok_weak` as well. Its training scripts are therefore 81% oracle-correct, and 15% use EPSG:3857 (35/237 in replicate 0), against 43% (191/444) for §2.5's `sft_base_weak`.

  E1 therefore tests how the gate ranks candidates within tasks the model can sometimes solve cleanly. §2.5 tested task-level rejection and candidate filtering together, on different task sets (`sft_base_meta` 147 tasks, `sft_base_weak` 444). Neither improves GSB v2: +0.9 pp in §2.5, −0.0 pp here.
- **How little the distillation moves the model.** Training is 237 samples × 2 epochs, 30 optimizer steps. Against the bare model on the same server, the LoRAs change the Protocol A outcome on 5.5% (`bok_meta`) and 4.7% (`bok_weak`) of test tasks. §2.5 measured 4.9% for two temperature-0 repeats of the bare model.
- **On OSM, descriptively, `bok_meta` is ahead.**
  - The difference is +5.4 pp (4/6 replicates, p = 0.16), and `bok_meta` has 7.5 pp fewer silent failures than `bok_weak`.
  - `bok_weak` sits at the bare model's level.
  - The direction matches §2.5 (`sft_base_meta − sft_base_weak` +16.7 pp on the same OSM set), but the effect is smaller and not established. Part of the shrinkage comes from the T restriction: in §2.5 `sft_base_weak` fell below the bare model on OSM (51.0% against 54.2%), while here `bok_weak` does not (56.6% against 57.3%).
- **Against §2.5** (descriptive, different server): the E1 arms (31.6%) sit where `sft_base_weak` (31.6%) and `sft_base_meta` (32.4%) did on Protocol A. The bare model itself scores higher on the A100 than on the G4 (30.9% against 29.0% on Protocol A, 57.3% against 54.2% on OSM), which is why E1 evaluates it on the same server. Against that same-server baseline the E1 arms gain about +0.7 pp; in §2.5 `sft_base_meta` gained +3.5 pp [−0.1, +7.9] over its bare model.

---

## 3. Open Methodological Risks & Mitigation Tracker

| ID | Risk / Reviewer Critique | Current Status | Concrete Mitigation |
|---|---|---|---|
| **R1** | **Target-location overlap**: `metamorphic.py` relocates local inputs to `(58.3°, 6.0°)` / `(-56.2°, 12.0°)`, overlapping GSB v1 `high`/`zone_edge`. | **Measured (2026-11-26): position yes, mechanism no** | §2.6.2–3 (pre-registered, descriptive). `arch_coev`'s per-variant gain over `naive` on blind positions (+22.4 pp, 6/6 seeds) matches its gain on the designed variants (+18.8) and holds ≥ 15° from every probe (+21.2). It shrinks to +7.7 pp beyond the UTM limits. Most of it is shared with `coev_tip_only` (+18.8), which uses the same 14 generations. On held-out storage mechanisms, no method beats `naive`: near +4.3 [−5.5, +14.1], and the far comparison is uninformative. Earlier evidence: polar Protocol A/B on GSB v1 (Demonstrated self-certified polar verifier discovery breaking Polar-A (`0.2% → 85.9%`) and Polar-B (`13.6% → 82.1%`) with zero polar hints or labels; expanding to polar + antimeridian + raster/OSM in GSB v2.) |
| **R2** | **Diagnostic hint leakage in v0.2**: `evaluator.py` previously appended a GIS/CRS hint string inside `meta_diag`. | **Resolved (2026-11-21)** | Verified `--minimal-meta-prompt` + anonymized `probe_1/probe_2` AND zero-diagnostic `meta_gate_only` (`80.8%`, 3/3 seeds > `darwinx_soft` `76.9%`). |
| **R3** | **Missing baselines & seed count**: Need non-evolution baselines, Oracle/soft-DarwinX/LLM-judge controls, and ≥6–8 seeds. | **6 seeds on one server (2026-11-26); confirmatory comparison not established** | §2.6.1: `arch_coev − naive` = +6.7 pp on Protocol A (5/6 seeds, sign-flip p = 0.25). Under Addendum 2, seeds are not added after seeing this result. A new test of the difference needs a new pre-registration with more seeds. |
| **R4** | **Swallowed-exception degradation**: Rule-heavy prompts and OOD fallbacks silently swallow exceptions. | **Resolved & ablated (2026-11-21)** | Domain-neutral `sys.monitoring` telemetry broke 2nd-order `except RuntimeError: EPSG:3857` lock-in during polar co-evolution (`16.7% → 88.9%` in-loop). |
| **R5** | **Bounded iteration vs true RSI**: Fixed human-written `metamorphic.py` + closed-API Meta-Agent. | **Weakened (2026-11-25)**: §2.5 attributes the weight-level gains to a single strong teacher; weak-filter SFT and weak-reward GRPO match the metamorphic versions. The GSB v2 probes were certified by a closed model. | Verified (a) non-circular self-certified verifier co-evolution, (b) 100% open-weight symmetric `26B → 26B` evolution, and (c) Dual-G4 Two-Timescale Harness → SFT + GRPO-Metamorphic → Gen-2 compounding (`36.5% → 93.3%`). |
| **R6** | **Gate false rejections**: the metamorphic gate assumes every probe preserves the answer. | **Measured and traced (2026-11-25), open** | Section 2.4: the full gate rejects 5.8–20.4% of oracle-correct GSB v2 scripts. 15 of the 96 rejections are gate errors: a probe that changes the reference answer (6), malformed antimeridian geometry from the polar probe (8), and a tolerance tighter than the oracle's (1); the rest fail on answer-preserving inputs the oracle does not test (mixed CRSs, km or projected storage, polar scenes). On real OSM, correct absolute-position solutions are rejected (latitude 8/8). Next: split relocated geometries at the antimeridian, widen the agreement tolerance to about twice the oracle's, skip relocating probes for absolute-position answers, use sum- or majority-preserving resampling for thresholded rasters, and report the breakdown next to every gate result. |
| **R7** | **Teacher vs. filter confound in weight internalization**: `SFT-Weak` and `SFT-Meta` used different teachers. | **Tested (2026-11-27): no added value on GSB v2** | §2.5: with the same teacher the filter makes no difference (Protocol A +2.0, B +0.0 pp). E1 (§2.7, pre-registered, teacher-free best-of-K, 6 replicates): the gate makes the data cleaner (+26.4 pp pool precision, p = 0.031), but `bok_meta − bok_weak` on Protocol A is −0.0 pp (p = 1.0). Both arms stay within 0.7 pp of the bare model. The claim that the filter improves training data is dropped for GSB v2. On OSM the difference is +5.4 pp, descriptive. |

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
- [x] **Two-Timescale Weight Internalization on GSB v2 (Wave S4-2)** *(done 2026-11-25)*: Completed disjoint `train[20:60]` `SFT-Weak` vs `SFT-Meta` + 24-step alternating `GRPO-Metamorphic` (3 certified probes, `0/768` `EPSG:3857`) on Colab G4 96 GB: bare `seed.py` climbs `30.7% → 56.2% → 62.4%` (`SFT-Weak = 30.4%`), Gen-2 zero-GIS-rule compounding reaches **`68.2%` ALL / `69.3%` HO** (`+5.3 pp` / `+9.3 pp` over Teacher) and **`100.0%` real OSM** (`GRPO-Meta` bare `91.7%` vs `SFT-Weak` `12.5%`).

### Phase P2 — External Generalization & Publication
- [x] **6-Seed GSB v2 Matrix + Archive-Preserving Co-Ev + 6-Seed OSM** *(Wave 1A/1B/1C done 2026-11-25)*: Completed 6 seeds each of `darwinx_soft`, `meta_T0`, `llm_judge`, `coev_tip_only`, `arch_coev` (`41.0 ± 13.6%`, `+17.4 pp` vs `meta_T0` $p=0.0207$, 5/6 breakout; *superseded by the unified re-evaluation, §2.6: +6.7 pp vs `naive`, p = 0.25, and the `meta_T0` comparison is descriptive*), 6-seed real OSM transfer (`86.1 ± 26.2%`, 5/6 at `91.7%–100.0%`), plus `runtime_meta`, `naive`, `meta_gate_only`, `oracle_soft`.
- [x] **Non-GIS Cross-Domain Generalization (`SciUnits`, Wave 1C)** *(done 2026-11-25)*: Completed 3-seed paired symmetric `26B → 26B` (`7.1%` vs `7.1%`, Meta-Agent synthesis bottleneck) and `gemini-4-hydrogen → 26B` (`darwinx` 3/3 locked at `7.1% / 91.4% Lucky` vs `meta_soft` `10.4 ± 7.5%`, `s2 = 18.2% ALL / 13.3% HO / 0.0% Err`, revealing the isomorphic static tabular verifier coverage gap).
- [x] **External Benchmark Transfer (`GeoAnalystBench`, Wave S4-2)** *(done 2026-11-25)*: Audited all 4 weights at `max_tokens=4096` (`100% syntax_ok`): `SFT-Meta` / `GRPO-Meta` lift `dynamic_crs_vec` from **`0.0% → 71.4%`** and `strict_geo_sound` from `76.5% → 88.2% / 82.3%`, whereas `SFT-Weak` degrades to `64.7%` (`raster_nodata_safe 75.0% → 37.5%`).

---

## 5. Local Run & Artifact Registry (Gitignored Data)

- `data/v1/`: 2,700 train + 480 test tasks (`tasks.jsonl` + variant directories).
- `data/v2/`, `data/osm_eval/`, `data/sci_v1/`: 1,950 GSB v2 tasks, 24 live 4-city OSM tasks, and 760 non-GIS SciUnits tasks.
- `results/zero_shot_v1/`: 5 zero-shot baselines (`e4b_think`, `e4b_think_hint`, `e4b_think_rules`, `q27_low`, `g26b_think`), rule & `gemini-decision-1` error labels, `caught_{base,rules}.json`.
- `runs/v3_gis_s{0..3}_{naive,preserve,meta}/`, `runs/eval480/`, `runs/eval_ood/`: Canonical v0.2 4-seed archive and full-480 / Polar-440 evals.
- `runs/v4_jitter_check.json`, `runs/v4_split_2g4_bench.json`: BF16 jitter floor and Dual-G4 split feasibility measurements.
- `runs/v4_block1_evals/`, `runs/v4_wave1_evals/`, `runs/v4_coev_polar/`, `runs/v4_sym26b_evals/`, `runs/v4_sym26b_harnesses/`, `runs/v4_grpo/`: Complete v4 ablation, co-evolution, symmetric 26B, and E4B GRPO artifacts (all 45 JSONs `usage.errors == 0`).
- `runs/v5_two_timescale/`: Complete A3 Dual-G4 26B Two-Timescale RSI artifacts — disjoint `train[20:55]` trajectories, `sft_weak` / `sft_meta` / `grpo_meta_final` summaries & full `eval480` JSONs, Gen-2 archive & `gen2_best_harness.py`, and trained bf16 LoRA checkpoint (`lora_grpo_meta_final/`, all 4 eval JSONs `usage.errors == 0`).
- `runs/v6_gsb_v2_s3/`, `runs/v6_gsb_v2_s3b/`, `runs/v6_gsb_v2_s3b_stage2/`: GSB v2 single-seed 4-tier ladder, 3 certified probes, and live OSM baseline evaluations.
- `runs/v7_s4_wave1{a,b,c}_g4{a,b}/`, `runs/figures/`: Complete audited Sprint 4 Wave 1A/1B/1C 6-seed GSB v2, 6-seed OSM, 3-seed `SciUnits` (symmetric + `gem4h`), and regenerated publication tables & SVG figures (`summary_tables.md`, `summary_data.json`, `fig_gsb2_multiseed.svg`, `fig_gsb2_two_timescale.svg`).
- `runs/v9_rebuttal_gpu/`: rebuttal controls (§2.5): trajectories and summaries for `same_teacher_weak` / `base_weak` / `base_meta`, LoRA configs for the 5 new adapters, Protocol A (with `.code.jsonl` scripts) and Protocol B GSB v2 evals, 8-city and 4-city OSM evals, and `missing_6seed/` (seeds 1–5 of `naive`, `meta_gate_only`, `oracle_soft`: archives, best harnesses, evals); all 52 eval JSONs `usage.errors == 0` with per-task records.
- `runs/v10_unified_6seed/`: the unified one-server re-evaluation (§2.6). `harnesses/` (48), `manifest.json`, and `evals/`: 144 eval JSONs (48 × `A_rep1` / `A_rep2` / `B_rep1`), the Protocol A `.code.jsonl` scripts, and 96 `.blind.json` and 96 `.heldout.json` sidecars. Summaries: `unified_summary_ref_{naive,arch_coev}`, `blind_summary` and `heldout_summary` (`.md` / `.json`).
- `runs/v11_gate_precision/`: gate-vs-oracle audits (§2.4): `gate_vs_oracle_*.jsonl` per trajectory source, the reruns with the E1 probes at 2% and 1% tolerance (`*_e1*`), `false_rejection_causes*`, and the OSM reference check.
- `runs/v12_e1_bok/`: E1 (§2.7). `e1_scored_candidates.jsonl` (600 tasks × 16 candidates), `arms/` (12 arm files and `e1_selection_audit.json`), 12 LoRA directories with training summaries, `evals/` (39 evals with `.code.jsonl`), and `e1_summary.{md,json}`.
- `runs/v8_s4_wave2_v2/`: Complete Sprint 4 Wave S4-2 GSB v2 Two-Timescale RSI artifacts — disjoint `train[20:60]` `weak`/`meta` trajectories, bf16 LoRA checkpoints (`sft_weak/`, `sft_meta/`, `grpo_meta_final/`), 24-step GRPO summary, bare-`seed.py` & Gen-2 compounding `eval450` + real OSM JSONs, and 4-weight `GeoAnalystBench` (`4096` tok) audit JSONs (all 71 Sprint 4 JSONs strictly audited `usage.errors == 0` & `calls >= n`).

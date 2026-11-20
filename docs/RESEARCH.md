# `meta-rsi` — Long-Term Research Plan, Experiment Log & Roadmap

> Living master document for the `meta-rsi` research project, undergraduate thesis, and SCI / top-venue manuscript.
> Repository: `https://github.com/Konakonai/meta-rsi`
> Last updated: 2026-11-20.
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

## 3. Open Methodological Risks & Mitigation Tracker

| ID | Risk / Reviewer Critique | Current Status | Concrete Mitigation |
|---|---|---|---|
| **R1** | **Target-location overlap**: `metamorphic.py` relocates local inputs to `(58.3°, 6.0°)` / `(-56.2°, 12.0°)`, which overlaps GSB v1 `high` (`±52–64°`) and `zone_edge` (`6°` grid). | **Full-440 Polar OOD complete (2026-11-20); GSB v2 & P1 verifier next** | Completed full 440-task Polar OOD stress test across all 13 harnesses (`runs/eval_ood/`): proved `metamorphic` fails loudly (`83.7%` Error, `2.7%` SilW) instead of silently (`preserve` `42.5%` silent), uncovered the hardcoded-UTM polar cylinder pitfall, and established the concrete need for polar/antimeridian AEQD operators in GSB v2 & Phase P1. |
| **R2** | **Diagnostic hint leakage in v0.2**: `evaluator.py` previously appended a GIS/CRS hint string inside `meta_diag` on metamorphic mismatch. | **Code fixed (2026-11-20), ablation pending** | `evaluator.py` is now **100% domain-neutral by default** (`include_domain_hint=False`); pass `--meta-domain-hints` only for explicit ablation (`meta_neutral` vs `meta_domain_hint`). |
| **R3** | **Missing gate baselines & seed count**: Need Oracle upper bound, LLM-judge gate, and ≥8 seeds on all 480 test tasks for publishable power. | **Full-480 complete for 4 seeds (2026-11-20); `--gate oracle` ready, `llm_judge` pending** | Run `--gate oracle` and a `gemini-decision-1` rubric gate alongside `naive`/`preserve`/`metamorphic`; expand to ≥8 seeds on GSB v2. |
| **R4** | **Swallowed-exception degradation**: Rule-heavy prompts caused 26% of runnable-but-wrong scripts to catch runtime exceptions and print `0`. | **Telemetry run on Polar OOD + `swallowed_runnable_count` added (2026-11-20)** | Log `dev_swallowed_rate` during evolution and `swallowed_runnable_count` on OOD/test evals to verify evolved harnesses do not hide exceptions. |
| **R5** | **Bounded iteration vs true RSI**: Fixed human-written `metamorphic.py` + fixed Meta-Prompt could be viewed by ICML/NeurIPS reviewers as bounded harness search rather than true RSI. | **Planned (Phase P1)** | Implement self-discovering/co-evolving invariants, HyperAgents-style metacognitive self-modification, and Two-Timescale Harness → Weight internalization. |

---

## 4. Phased Technical Roadmap

### Phase P0 — Immediate Methodological Rigor & Controls
- [ ] **Literature Review Package**: Deep-read full texts of `DarwinX`, `Hyperagents`, `When Self-Evolution Backfires`, `GISAgentBench`, `GeoAnalystBench`, `GeoContra`; keep [LITERATURE.md](LITERATURE.md) up to date as the Related Work backbone.
- [ ] **Neutral-Diagnostic Ablation (R2)**: Run paired seeds of `metamorphic` with default domain-neutral `meta_diag` vs `--meta-domain-hints` to verify Meta-Agent diagnoses CRS drift purely from numeric inconsistency + code inspection.
- [x] **Complete v0.2 Evaluation Matrix** *(done 2026-11-20)*: Finished full 480-task test eval (`runs/eval480/`, 13/13 `usage.errors == 0`) + full 440-task polar OOD stress test (`runs/eval_ood/`, 13/13 `usage.errors == 0`, `--track-caught`) on Gen-0 seed + all 12 `v3_gis` harnesses; updated `analysis.py`, SVG figures (`fig_rsi_gates.svg`, `fig_polar_ood.svg`), and docs.
- [ ] **Upper-Bound & LLM-Judge Gate Controls (R3)**: Run `--gate oracle` baseline + implement & run `gemini-decision-1` rubric gate (`--gate llm_judge`).
- [ ] **GSB v2 & Real OSM Validation Slice**:
  - Add polar (`>84°N / <80°S`), antimeridian (`±180°`), decoupled primary keys (`home_id`/`facility_id` vs duplicate `id`), and non-projection held-out traps (raster zonal stats with `nodata`, one-to-many `sjoin` deduplication).
  - Build a 20–30 task **real OpenStreetMap urban validation slice** across 3 latitudes (Singapore, Xuzhou, Helsinki) with map overlay figures comparing `EPSG:3857` vs proper projection buffers.

### Phase P1 — True RSI Depth (Beyond Bounded Harness Iteration)
- [ ] **Self-Discovering / Co-Evolving Metamorphic Verifier**: When in-loop accuracy saturates or `LARGE_EXTENT` stagnates, spawn a Verifier/Diagnostic Agent to autonomously propose/select candidate domain-preserving perturbations (polar relocation, scale/unit checks, CRS storage flip, schema/column permutation, raster nodata perturbation) and validate their invariance on seed sanity checks.
- [ ] **Metacognitive Self-Modification (HyperAgents-style)**: Let the Meta-Agent co-evolve its own diagnosis/mutation policy and archive selection parameters alongside `harness.py`, measuring cross-generation meta-gain.
- [ ] **Two-Timescale RSI (Harness → Weight Internalization)**: Distill verified trajectories from the evolved Harness back into `gemma-4-E4B-it` weights via LoRA SFT / GRPO on Colab G4 (using timing & gotcha configs in [INFRA_AND_GOTCHAS.md](INFRA_AND_GOTCHAS.md)), prune prompt rules, and run Generation-2 Harness evolution on the internalized weights.

### Phase P2 — External Generalization & Publication
- [ ] **Scale to ≥8 Paired Seeds** across all gates on GSB v2 (full test set).
- [ ] **Cross-Model Zero-Shot Transfer**: Evaluate E4B-evolved harnesses zero-shot on `gemma-4-26B-A4B-it`, `Qwen3.8-27B`, and Gemini 4.
- [ ] **External Benchmark Transfer**: Evaluate evolved harnesses on Python analysis subsets of `GeoAnalystBench` (`arXiv:2509.05881`) and `GISAgentBench` (`arXiv:2608.01645`).
- [ ] **Broader Geospatial & Non-GIS Generalization**:
  - Controlled **Google Earth Engine (GEE)** / raster (`rasterio`) task subset.
  - Non-saturated general Python/tabular benchmark (`benchmarks/py_v2/`, target Gen-0 `OraDev < 35%, Lucky > 30%`) to demonstrate domain-agnostic generality.

---

## 5. Local Run & Artifact Registry (Gitignored Data)

- `data/v1/`: 2,700 train + 480 test tasks (`tasks.jsonl` + variant directories).
- `results/zero_shot_v1/`: 5 zero-shot baselines (`e4b_think`, `e4b_think_hint`, `e4b_think_rules`, `q27_low`, `g26b_think`), rule & `gemini-decision-1` error labels (`errors_r2.json`, `errors_llm_v2.jsonl`), `caught_{base,rules}.json`, and blind spot-checks.
- `runs/v3_gis_s{0..3}_{naive,preserve,meta}/`: Canonical v0.2 4-seed evolution trajectories.
- `runs/eval480/*.json`: **Complete full-480 unseen test matrix** (13/13 JSONs verified `n=480, usage.errors == 0`).
- `runs/eval_ood/*.json`: **Complete full-440 Polar OOD stress-test matrix** (13/13 JSONs verified `n=440, usage.errors == 0`, `--track-caught`).
- `runs/eval150/*.json`: Earlier 150-task stratified subset (13/13 JSONs verified `usage.errors == 0`, archived for reference).
- Earlier diagnostic runs (archived locally for reference only — do not mix into `v3_gis` aggregates): `local_pilot_meta`, `g4_e4b_*` (v0.1 `temp=0.2, dev=18`), `v2_gis_s0_*` (pre-scaling-fix), `v3_py_s0_*` (saturated `py_v1` draft).

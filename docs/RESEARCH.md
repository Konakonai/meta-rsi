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
- **Unseen Test Set (`runs/eval150/`, all 13 JSONs verified `usage.errors == 0`)**: Stratified **150 unseen 3-location tasks** (90 seen + 60 held-out = 450 variant executions/harness).

### Main 4-Seed Aggregate (`mean ± std`, `n = 150` unseen tasks)

| Method | Seeds | Dev `Low` (Weak) | Dev `Oracle` (3-loc) | Dev Gap | **Test ALL (`n=150`)** | **Test Seen (`n=90`)** | **Test Held-out (`n=60`)** | **Test `Lucky`** | Test `Err` | Calls | Lock-In (Gap ≥15%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gen-0 Seed (`harnesses/seed.py`) | 1 | 45.8% | 17.6% | +28.2% | 28.0% | 24.4% | 33.3% | 31.3% | 17.3% | 1.00 | — |
| `naive` (hill-climbing) | 4 | 84.7 ± 2.8% | 66.7 ± 24.3% | 18.1 ± 23.6% | 70.0 ± 23.6% | 67.5 ± 17.7% | 73.8 ± 32.6% | 20.3 ± 21.7% | 2.7 ± 1.2% | 1.26 | 1/4 |
| `preserve` (DarwinX-style) | 4 | 75.7 ± 11.9% | 50.7 ± 21.9% | 25.0 ± 12.6% | 54.5 ± 20.9% | 52.5 ± 17.4% | 57.5 ± 26.6% | 23.8 ± 12.7% | 5.2 ± 1.9% | 1.27 | **3/4** |
| **`metamorphic` (Ours)** | 4 | 80.6 ± 2.3% | **77.8 ± 3.9%** | **2.8 ± 5.6%** | **81.8 ± 3.3%** | **75.8 ± 3.1%** | **90.8 ± 7.4%** | **2.3 ± 2.5%** | 7.5 ± 2.6% | 1.47 | **0/4** |

- **Paired seed comparisons**: `metamorphic` beats `preserve` on 4/4 seeds (`+27.3 ± 23.3%` Test ALL, `p=0.1006`) and reduces `Lucky` on 4/4 seeds (`-21.5 ± 13.5%`, paired $t=-3.19, p=0.0496$); beats `naive` on `Lucky` on 4/4 seeds (`-18.0 ± 22.5%`) and cuts cross-seed std by **6.3×–7.2×** (`3.3%` vs `20.9%`/`23.6%`).
- **4B No-Think + Evolved Harness vs 26B/27B Thinking**: Best evolved harness (`runs/v3_gis_s2_meta/gen_003/harness.py`, `1.27` calls/task) reaches **86.7% ALL / 98.3% Held-out / 0.7% lucky**, surpassing zero-shot thinking `gemma-4-26B-A4B-it` (`71.7%`) and `Qwen3.8-27B` (`62.2%`) (see [GSB_V1_BASELINE.md](GSB_V1_BASELINE.md)).

### Task Category Breakdown (`n = 150`, `correct% / lucky%`)

Reproducible anytime via `python -m meta_rsi.eval.analysis runs/eval150`:

| Method | `LOCAL_PROJ` (15 families) | `NON_PROJ_TRAP` (4 families) | `LARGE_EXTENT` (2 families) |
|---|---:|---:|---:|
| Gen-0 Seed | 25.8% / 35.8% | 35.0% / 10.0% | 40.0% / 20.0% |
| `naive` | 73.3 ± 30.0% / 22.5% | 50.0 ± 15.8% / 12.5% | 70.0 ± 29.4% / 10.0% |
| `preserve` (DarwinX) | 55.0 ± 26.5% / 28.1% | 51.2 ± 12.5% / 6.2% | 55.0 ± 5.8% / 7.5% |
| **`metamorphic` (Ours)** | **89.2 ± 3.8% / 1.2%** | **53.8 ± 11.1% / 7.5%** | 50.0 ± 8.2% / 5.0% |

Key diagnostic insights from this breakdown:
1. **Zero skill contamination on non-projection tasks**: Unlike hand-written CRS rules (`e4b_think_rules`, which collapsed `points_in_polygon` `78.7% -> 41.9%` and `area_weighted_population` `43.8% -> 8.7%`), `metamorphic` *improves* `NON_PROJ_TRAP` (`35.0% -> 53.8%`, highest of all 3 gates).
2. **Principled headroom on `LARGE_EXTENT`**: Our current large-extent isometry (longitude shift + latitude mirror) preserves single-UTM-zone distortion when the UTM zone tracks the shifted centroid, explaining why `LARGE_EXTENT` stays at `50.0%` — motivating **self-discovered / stronger invariance operators** in Phase P1.

### Partial Full-480 Unseen Test Runs (`runs/eval480/`)
- `seed_e4b_nothink.json`: `ALL=24.2%` (`lucky=31.2%`, `silent_wrong=26.5%`, `error=18.1%`), `SEEN_18=23.1%`, `HELDOUT_3=27.5%`.
- `v3_gis_s0_naive.json` (overfitted seed): `ALL=32.9%`, **`lucky=50.8%`** (worse than seed!), **`HELDOUT_3 lucky=74.2%`** (`correct=20.0%`).
- `v3_gis_s2_naive.json`: `ALL=75.2%` (`lucky=10.8%`, `error=3.5%`), `SEEN_18=71.1%`, `HELDOUT_3=87.5%`.
- Remaining 10 harnesses from `v3_gis_s{0..3}` still need full-480 evaluation (queued in P0).

---

## 3. Open Methodological Risks & Mitigation Tracker

| ID | Risk / Reviewer Critique | Current Status | Concrete Mitigation |
|---|---|---|---|
| **R1** | **Target-location overlap**: `metamorphic.py` relocates local inputs to `(58.3°, 6.0°)` / `(-56.2°, 12.0°)`, which overlaps GSB v1 `high` (`±52–64°`) and `zone_edge` (`6°` grid). | **Tooling ready (`eval_ood.py`), runs pending** | Evaluate all evolved harnesses on **disjoint polar OOD locations** (`86.5°N`, `-85.5°S`, where `estimate_utm_crs()` raises `RuntimeError` — verified 38/38 in `tests/test_meta_rsi.py`) via `python -m meta_rsi.eval.eval_ood`, add antimeridian (`±180°`) and non-projection traps in GSB v2, and validate on real OSM city slices. |
| **R2** | **Diagnostic hint leakage in v0.2**: `evaluator.py` previously appended a GIS/CRS hint string inside `meta_diag` on metamorphic mismatch. | **Code fixed (2026-11-20), ablation pending** | `evaluator.py` is now **100% domain-neutral by default** (`include_domain_hint=False`); pass `--meta-domain-hints` only for explicit ablation (`meta_neutral` vs `meta_domain_hint`). |
| **R3** | **Missing gate baselines & seed count**: Need Oracle upper bound, LLM-judge gate, and ≥8 seeds on all 480 test tasks for publishable power. | **`--gate oracle` implemented (2026-11-20); LLM-judge gate pending** | Run `--gate oracle` and a `gemini-decision-1` rubric gate alongside `naive`/`preserve`/`metamorphic`; expand to ≥8 seeds and full 480-task evals. |
| **R4** | **Swallowed-exception degradation**: Rule-heavy prompts caused 26% of runnable-but-wrong scripts to catch runtime exceptions and print `0`. | **Telemetry wired (`--track-caught-exceptions`, 2026-11-20)** | Log `dev_swallowed_rate` during evolution and OOD evals to verify evolved harnesses do not hide exceptions. |
| **R5** | **Bounded iteration vs true RSI**: Fixed human-written `metamorphic.py` + fixed Meta-Prompt could be viewed by ICML/NeurIPS reviewers as bounded harness search rather than true RSI. | **Planned (Phase P1)** | Implement self-discovering/co-evolving invariants, HyperAgents-style metacognitive self-modification, and Two-Timescale Harness → Weight internalization. |

---

## 4. Phased Technical Roadmap

### Phase P0 — Immediate Methodological Rigor & Controls
- [ ] **Literature Review Package**: Deep-read full texts of `DarwinX`, `Hyperagents`, `When Self-Evolution Backfires`, `GISAgentBench`, `GeoAnalystBench`, `GeoContra`; keep [LITERATURE.md](LITERATURE.md) up to date as the Related Work backbone.
- [ ] **Neutral-Diagnostic Ablation (R2)**: Run paired seeds of `metamorphic` with default domain-neutral `meta_diag` vs `--meta-domain-hints` to verify Meta-Agent diagnoses CRS drift purely from numeric inconsistency + code inspection.
- [ ] **Complete v0.2 Evaluation Matrix**: Finish full 480-task test eval (`eval_test.py`) + polar OOD stress test (`eval_ood.py`) on all 12 `v3_gis` harnesses (`--track-caught`).
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
- `runs/v3_gis_s{0..3}_{naive,preserve,meta}/` + `runs/eval150/*.json`: Canonical v0.2 4-seed dataset (`usage.errors == 0` verified across all 13 JSONs).
- `runs/eval480/`: Full 480-task evals completed so far (`seed_e4b_nothink`, `v3_gis_s0_naive`, `v3_gis_s2_naive`).
- Earlier diagnostic runs (archived locally for reference only — do not mix into `v3_gis` aggregates): `local_pilot_meta`, `g4_e4b_*` (v0.1 `temp=0.2, dev=18`), `v2_gis_s0_*` (pre-scaling-fix), `v3_py_s0_*` (saturated `py_v1` draft).

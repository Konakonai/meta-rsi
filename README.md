# meta-rsi

**Metamorphic-Gated Recursive Self-Improvement (RSI) for LLM Agents**, evaluated primarily on geospatial analysis tasks where weak verifiers admit *silent errors* (code that executes cleanly and passes local checks, yet fails catastrophically out-of-distribution).

## Core Idea

When self-evolving an Agent Harness against a weak in-loop verifier (such as single-location checks), standard archive policies like scalar hill-climbing (`naive`) or DarwinX-style `Preserve-and-Extend` (`preserve`) suffer from **Proxy Lock-In**: early mutations exploit location-specific shortcuts (e.g., measuring in Web Mercator `EPSG:3857` near the equator), and strict preservation gates then reject genuinely general fixes that drop even a single lucky task.

`meta-rsi` breaks Proxy Lock-In using **Label-Free Metamorphic Invariance Gating**:
- Given only a task's input workspace $x$ (zero extra labels or hidden test variants), an invariance operator $T(x)$ constructs an equivalent perturbed workspace (local AEQD spatial relocation, ellipsoidal isometry for large extents, `.gpkg` CRS storage flip, and row shuffling).
- A task counts as verified in-loop only if the emitted script passes the weak check **and** stays self-consistent between $x$ and $T(x)$.
- A **domain-neutral Meta-Agent** evolves a self-contained Python Agent Harness (`harness.py`) using only in-loop failures, domain-neutral metamorphic drift diagnostics (with optional `--meta-domain-hints` ablation), and a DGM/DarwinX lineage archive with recombination.

## Documentation

Long-term project documentation lives under `docs/`:
- [docs/RESEARCH.md](docs/RESEARCH.md) — Master research plan, verified experiment log, category breakdowns, open methodological risks, and phased roadmap.
- [docs/LITERATURE.md](docs/LITERATURE.md) — Living literature survey & positioning matrix (RSI/Harness evolution, GeoAI executable benchmarks, silent errors, spurious correctness).
- [docs/GSB_V1_BASELINE.md](docs/GSB_V1_BASELINE.md) — `benchmarks/gsb` v1 audit, 5-condition zero-shot results (`480 × 8`), rule + `gemini-decision-1` error taxonomy, and skill-contamination evidence.
- [docs/INFRA_AND_GOTCHAS.md](docs/INFRA_AND_GOTCHAS.md) — Compute inventory, budgets, Colab G4 vLLM & GRPO/LoRA benchmarks, and engineering runbook.

## Verified Results (v0.2, 4 Paired Disjoint Seeds)

- **Task LLM**: frozen `google/gemma-4-E4B-it` (`enable_thinking=False`, `temp=0.0`, max 4 calls/task)
- **Meta LLM**: `gemini-4-hydrogen` (`thinking_level=low`, domain-neutral system prompt)
- **Protocol**: 4 disjoint dev splits (36 tasks/seed = 2 per 18 seen families), 4 generations, evaluated on the **full 480 unseen 3-location test set** (360 seen + 120 held-out = 1,440 executions/harness, apples-to-apples with all zero-shot baselines) plus the **full 440-task Polar OOD stress test** (`>84°N / <80°S`).

| Method | Seeds | Dev `Low` (Weak) | Dev `Oracle` (3-loc) | Dev Gap | **Test ALL (`n=480`)** | **Test Seen (`n=360`)** | **Test Held-out (`n=120`)** | **Test `Lucky`** | Test `Err` | Calls | Lock-In (Gap ≥15%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gen-0 Seed (`harnesses/seed.py`) | 1 | 45.8% | 17.6% | +28.2% | 24.2% | 23.1% | 27.5% | 31.2% | 18.1% | 1.00 | — |
| `naive` (hill-climbing) | 4 | 84.7 ± 2.8% | 66.7 ± 24.3% | 18.1 ± 23.6% | 66.9 ± 22.7% | 64.7 ± 18.5% | 73.3 ± 35.7% | 18.6 ± 21.6% | 4.1 ± 1.7% | 1.30 | 1/4 |
| `preserve` (DarwinX-style) | 4 | 75.7 ± 11.9% | 50.7 ± 21.9% | 25.0 ± 12.6% | 52.1 ± 22.4% | 51.0 ± 20.5% | 55.4 ± 29.1% | 24.6 ± 15.4% | 7.4 ± 2.9% | 1.26 | **3/4** |
| **`metamorphic` (Ours)** | 4 | 80.6 ± 2.3% | **77.8 ± 3.9%** | **2.8 ± 5.6%** | **78.4 ± 3.8%** | **73.8 ± 4.9%** | **92.5 ± 5.1%** | **2.1 ± 1.8%** | 8.9 ± 3.0% | 1.50 | **0/4** |

- Best evolved 4B no-think harness (`v3_gis_s2_meta`, `1.29` calls/task) reaches **81.9% ALL / 99.2% Held-out (119/120) / 1.0% lucky**, outperforming zero-shot thinking `gemma-4-26B-A4B-it` (`71.7%` ALL / `13.4%` lucky) by **+10.2%** and `Qwen3.8-27B` (`62.2%`) by **+19.7%** on the exact same 480 tasks.
- On the **Polar OOD stress test** (`n=440` local tasks at `86.5°N / -85.5°S`, where `estimate_utm_crs()` raises `RuntimeError`), `metamorphic` fails **loudly** (`83.7%` Error, `2.7%` silent wrong) rather than returning silent distorted numbers (`Gen-0 Seed`: `65.9%` silent `Lucky+SilW`; `preserve`: `42.5%`), motivating self-discovered polar/AEQD operators in Phase P1 (full breakdown in [docs/RESEARCH.md](docs/RESEARCH.md)).

## Repository Layout

```text
meta-rsi/
├── meta_rsi/                   # Core RSI framework
│   ├── llm.py                  # Multi-backend LLM wrapper (openai-compat, gemini-cli, gemini-rest, mock)
│   ├── sandbox.py              # AST validator, dynamic harness loader, clean-copy sandbox & call cap
│   ├── metamorphic.py          # Label-free invariance operators T(x) (AEQD shift, isometry, CRS flip, tabular)
│   ├── evaluator.py            # Dual-view evaluator (in-loop weak verifier vs hidden 3-location oracle)
│   ├── archive.py              # Lineage archive, parent/recombination selection, gating policies
│   ├── meta_agent.py           # Domain-neutral Meta-Agent proposer
│   ├── evolve.py               # Main RSI evolution loop CLI
│   ├── harnesses/
│   │   └── seed.py             # Gen-0 seed Agent Harness (`solve(task_prompt, work_dir, llm_call, run_python)`)
│   ├── eval/                   # Evaluation & statistical reporting CLIs
│   │   ├── eval_test.py        # Unseen test-set evaluator (full 480 or stratified)
│   │   ├── eval_ood.py         # Polar OOD (>84N / <80S, UTM-undefined) stress-test evaluator
│   │   ├── report.py           # Side-by-side single-run summary
│   │   ├── stats.py            # Multi-seed aggregation & lock-in stats
│   │   └── analysis.py         # Paired t-tests, category breakdowns & publication tables
│   └── infra/                  # Cloud GPU / Colab G4 utilities
│       ├── colab_g4.py         # One-command Colab G4 bundle & launcher
│       └── start_g4_vllm.py    # Remote G4 vLLM server bootstrap
├── benchmarks/
│   └── gsb/                    # Geospatial Silent-Error Benchmark (21 families, 3-location oracle)
│       ├── geo.py              # Location variants & local-metre AEQD placement on WGS84
│       ├── families.py         # 21 GIS task families (18 seen + 3 held-out) & reference solvers
│       ├── build.py            # Dataset generator (`tasks.jsonl` + variant workspaces)
│       ├── harness.py          # Code extraction, 3-variant execution & scoring
│       └── audit.py            # Method separability audit (reference, UTM, EPSG:3857, fixed UTM)
├── scripts/                    # Zero-shot baseline, error-classification & plotting utilities
│   ├── zero_shot.py            # vLLM zero-shot baseline sampling & scoring
│   ├── classify_errors.py      # Rule-based silent-error fingerprinting
│   ├── classify_llm.py         # Gemini Decisions API error classification
│   ├── caught_exceptions.py    # Detect scripts that swallowed runtime exceptions
│   ├── check_aeqd_truth.py     # Verify AEQD vs ellipsoidal geodesic ground-truth agreement
│   └── plot_figures.py         # Generate SVG figures into docs/figures/
├── tests/                      # Unit & end-to-end tests (`test_gsb.py`, `test_meta_rsi.py`)
├── docs/                       # Living research docs, baseline audit & SVG figures
│   ├── RESEARCH.md             # Master research plan, experiment log & roadmap
│   ├── LITERATURE.md           # Unified literature survey & positioning matrix
│   ├── GSB_V1_BASELINE.md      # Complete GSB v1 zero-shot & silent-error report
│   ├── INFRA_AND_GOTCHAS.md    # Compute inventory, budgets & Colab G4 runbook
│   ├── gsb_v1_audit.log        # Method-separability audit log across all 21 families
│   └── figures/                # Generated SVG plots (`fig_outcomes`, `fig_mercator`, `fig_rsi_gates`, `fig_polar_ood`)
├── pyproject.toml              # Package metadata & pytest configuration
└── shell.nix                   # Reproducible NixOS geospatial Python environment
```

## Quick Start

Enter the reproducible geospatial Python environment (or install dependencies from `pyproject.toml`):

```bash
nix-shell
```

### 1. Build & Audit the Geospatial Silent-Error Benchmark (`benchmarks/gsb`)

```bash
python -m benchmarks.gsb.build --out data/v1 --train 150 --test 20 --heldout 40 --seed 0
python -m benchmarks.gsb.audit data/v1
```

### 2. Run RSI Harness Evolution

```bash
python -m meta_rsi.evolve \
    --exp my_meta_run \
    --gate metamorphic \
    --generations 4 \
    --dev-per-family 2 --dev-offset 0 \
    --task-backend openai-compat --task-url http://127.0.0.1:18000/v1 \
    --task-model google/gemma-4-E4B-it --task-temp 0.0 --task-no-think \
    --meta-backend gemini-cli --meta-model gemini-4-hydrogen \
    --meta-thinking-level low --meta-max-tokens 16384
```

### 3. Evaluate on Unseen Test Split, Polar OOD & Generate Reports / Figures

```bash
python -m meta_rsi.eval.eval_test \
    --harness runs/my_meta_run/gen_004/harness.py \
    --out runs/eval480/my_meta_run.json \
    --task-no-think

# Polar OOD stress test (>84N / <80S, disjoint from metamorphic targets, breaks estimate_utm_crs)
python -m meta_rsi.eval.eval_ood \
    --harness runs/my_meta_run/gen_004/harness.py \
    --out runs/eval_ood/my_meta_run.json \
    --task-no-think --seen-per-family 0 --heldout-per-family 0 --track-caught

python -m meta_rsi.eval.stats runs/v3_gis_s*
python -m meta_rsi.eval.analysis runs/eval480 --ood-dir runs/eval_ood

# Regenerate SVG figures into docs/figures/
nix-shell -p "python3.withPackages(ps:[ps.matplotlib ps.numpy])" --run "python scripts/plot_figures.py"
```

### 4. Run Tests

```bash
python -m pytest -q tests
```

## Geospatial Silent-Error Benchmark (`benchmarks/gsb`)

Each task places identical local geometry at three locations on the globe; a script must answer accurately on all three variants to count as `correct`.

| Variant | Center | Why |
|---|---|---|
| `low` | latitude within ±7° | `EPSG:3857` distortion < 0.75% in length (< 1.51% in area, < 1% within ±5.7°), so Web Mercator passes most `low` tasks by luck |
| `high` | latitude ±52–64° | `EPSG:3857` inflates lengths 1.6–2.3× (areas 2.5–5×) |
| `zone_edge` | latitude ±25–45°, on a UTM zone boundary | data straddles two zones; breaks any hard-coded UTM zone |

- **Ground truth**: Local tasks (extent < 30 km) use an AEQD projection centred on the bounding box (verified within 0.0001% of `pyproj.Geod` via `python -m scripts.check_aeqd_truth`). Large-extent tasks use `pyproj.Geod` on WGS84 directly. Prompts never state location or CRS; 25% of `.gpkg` layers are stored in a local UTM CRS, and point layers can be CSV with `lon`/`lat`.
- **Families (21 total)**:
  - *Seen (18)*: `buffer_union_area`, `count_within`, `mean_nn_distance`, `sum_within`, `nearest_facility_max`, `area_by_class`, `perimeter_by_class`, `overlay_area`, `centroid_distance`, `road_length_by_type`, `count_near_roads`, `roads_in_district`, `points_in_polygon`, `area_weighted_population`, `covered_population`, `mean_per_district`, `large_region_area`, `large_route_length`.
  - *Held-out (3)*: `density_in_polygon`, `road_buffer_area`, `max_pair_distance`.
- **Scoring**: Real answers within 1% of truth, counts exact. Categories: `correct` (all 3 right), `lucky` (runs everywhere, right on some variants only), `silent_wrong` (runs everywhere, wrong everywhere), `error` (exception, timeout, or non-numeric output), `no_code`.
- **Audit**: See [docs/gsb_v1_audit.log](docs/gsb_v1_audit.log) for method-separability verification across all 21 families.

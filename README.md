# meta-rsi

**Recursive self-improvement of LLM agent harnesses under weak verifiers, on executable geospatial analysis.**

A geospatial script can run cleanly, pass a local check, and still be wrong somewhere else on the Earth. A common case is measuring areas in Web Mercator (`EPSG:3857`): near the equator that is almost right, at 60° it is off by a factor of four. A self-improving agent loop that is checked only on easy locations learns such shortcuts and keeps them (*proxy lock-in*).

`meta-rsi` asks whether **label-free metamorphic checks** can stop this. A metamorphic check moves or re-encodes a task's input so the answer must stay the same, and asks the agent's script for the same answer on both versions. The project builds:
- the evolution loop and the gates;
- two silent-error benchmarks (GSB v1 and v2), plus blind and held-out-mechanism test sets;
- weight-level distillation (SFT, GRPO);
- the pre-registered analyses used to evaluate all of it.

## Status (2026-11-26)

The current evidence, from pre-registered analyses on a one-server re-evaluation of 48 evolved harnesses (8 methods × 6 seeds):

- **No method establishes a gain on the strict endpoint.** The pre-registered confirmatory comparison, archive-preserving co-evolution (`arch_coev`) against plain hill-climbing (`naive`), is +6.7 pp. That is 5/6 seeds, exact sign-flip p = 0.25.
- **Descriptively, the two co-evolution methods write scripts that are right on more relocated positions.** Per variant, `coev_tip_only` gains +18.8 pp and `arch_coev` +22.4 pp over `naive` on blind positions, on 6/6 seeds each. The `arch_coev` gain holds 15° or more away from every probe latitude (+21.2 pp), so it is not tied to the probes' coordinates.
- **That gain stays inside the failure mechanisms the probes exercise** (Web Mercator distortion, the UTM range).
  - It shrinks to +7.7 pp [−2.1, +17.5] beyond the UTM limits.
  - It is not established on storage conventions no probe writes (near transfer +4.3 [−5.5, +14.1]).
  - It costs more loud failures: 25% errors against 3%.
- **The gain comes with a larger search budget.** Both co-evolution methods searched 14 candidate generations, every other method 6. Their probes were certified once by a closed model. At 6 generations, the binary gate alone (`meta_gate_only`) gives +4.0 pp per variant, and a static LLM judge +15.3 pp.
- **Weight-level gains come from the teacher, not the filter.** With the same teacher, weak filtering matches metamorphic filtering. On bare-model data, the filter helps only on real OpenStreetMap tasks (+16.7 pp).
- **The gate rejects 6–20% of correct scripts**, depending on the trajectory source. 15 of 96 audited rejections are errors in the gate itself.
- **Next: E1**, a teacher-free best-of-K test of whether the filter adds value once the model can write candidates. Its code and analysis are committed before any data; it is waiting for a GPU.

Full tables, statistics and audit notes are in [docs/RESEARCH.md](docs/RESEARCH.md) §2.4–2.6.

## Method

**The loop.** A meta-agent evolves a self-contained Python harness (`harness.py`). The harness drives a frozen task LLM that writes and runs analysis scripts. Candidates enter a lineage archive (DGM / DarwinX-style, with recombination) if they pass the in-loop verifier. The meta-agent's prompt is domain-neutral: it sees in-loop failures and drift diagnostics, not GIS rules.

**The gate.** The weak in-loop verifier checks each dev task at one location (`low`, near the equator). The metamorphic gate also requires the same script to give the same answer on transformed copies of the input:
- **Built-in probes:** relocation to 58.3°N / 56.2°S with row shuffling, and a CRS storage flip.
- **Certified probes:** 3 further probes proposed and certified by a verifier agent (polar LAEA relocation to 86°N, a Web Mercator–stored GeoPackage, km-unit storage).

No labels or hidden test variants are used.

**Methods compared on GSB v2** (6 seeds each, `gemma-4-26B-A4B-it` as task and meta model):

| Method | In-loop verifier | Generations |
|---|---|---:|
| `naive` | weak check, hill-climbing | 6 |
| `darwinx_soft` | weak check, soft regression gate | 6 |
| `meta_gate_only` | weak check + metamorphic gate, no diagnostic text | 6 |
| `oracle_soft` | 3-location oracle labels (upper-bound control) | 6 |
| `meta_T0` | one fixed mid-latitude probe | 6 |
| `llm_judge` | static LLM code-review rubric | 6 |
| `coev_tip_only` | Stage 2 co-evolution with the 3 certified probes, from the Stage-1 tip | 6 + 8 |
| `arch_coev` | the same, keeping the Stage-1 archive (archive-preserving co-evolution) | 6 + 8 |

## Main result: GSB v2, 6 seeds, one server

All 48 harnesses ran on one Colab G4 vLLM server (bf16, temperature 0). Protocol A ran twice on the 450 test tasks: one script is written on `low` and must be right on every variant. All 144 evals are clean. The decision rule was fixed before any comparison ([docs/BLIND_PROTOCOL.md](docs/BLIND_PROTOCOL.md), Addenda 1–4):
- there is one confirmatory comparison, established only if all 6 seeds agree (exact sign-flip p < 0.05);
- everything else is descriptive.

| Method | Correct | Silent failures | Errors | − `naive` (pp) [t₅ 95%] | Sign-flip p |
|---|---:|---:|---:|---|---:|
| `naive` | 33.8 ± 5.6% | 63.1% | 3.0% | — | — |
| `darwinx_soft` | 31.2 ± 1.4% | 63.9% | 4.9% | −2.6 [−8.5, +3.2] | 0.31 |
| `meta_T0` | 23.3 ± 5.4% | 35.4% | 41.3% | −10.5 [−19.5, −1.5] | 0.062 |
| `meta_gate_only` | 36.6 ± 7.7% | 55.6% | 7.7% | +2.8 [−0.7, +6.3] | 0.12 |
| `coev_tip_only` | 37.5 ± 18.1% | 35.3% | 27.1% | +3.7 [−11.8, +19.2] | 0.53 |
| `oracle_soft` | 39.4 ± 10.7% | 53.1% | 7.5% | +5.6 [−2.2, +13.4] | 0.16 |
| `llm_judge` | 39.9 ± 15.0% | 42.6% | 17.4% | +6.1 [−12.1, +24.3] | 0.38 |
| **`arch_coev`** (confirmatory) | 40.5 ± 13.3% | 34.1% | 25.4% | **+6.7 [−5.0, +18.3]** | **0.25** |

Silent failures are `lucky` (right on some variants only) plus `silent_wrong` (wrong everywhere).

- **The unified eval reproduces the original per-method evals within 0.8 pp.** Repeat noise is 1.3 pp between Protocol A repeats.
- **Seed 0 was favourable for the evolved harnesses themselves.** Its gap to seeds 1–5 is +16.7 pp on average, so earlier single-seed comparisons are one seed out of six.
- **Protocol B** (a new script per variant, 100 tasks) gives `arch_coev − naive` = +8.7 pp [−3.8, +21.1] (descriptive).

## Where the gain is, and where it is not (descriptive)

The same saved Protocol A scripts were re-run offline:
- on **blind relocations**: 8 positions per task, drawn area-uniformly in four latitude bands without reference to the probes;
- on **held-out storage mechanisms**: the `low` scene stored in a convention no probe writes.

Values are the share of variants answered correctly.

| Test set / stratum | `naive` | `coev_tip_only` | `arch_coev` | `arch_coev − naive` (pp) [t₅ 95%] |
|---|---:|---:|---:|---|
| Designed variants (`high`, `polar`, antimeridian) | 40.7 | 56.0 | 59.5 | +18.8 [+8.2, +29.3] |
| Blind, all positions | 43.0 | 61.8 | 65.4 | +22.4 [+11.7, +33.2] |
| Blind, ≥ 15° from every probe latitude | 43.7 | 67.1 | 64.9 | +21.2 [+11.2, +31.2] |
| Blind, UTM-valid zone | 36.6 | 71.6 | 69.9 | +33.3 [+17.5, +49.1] |
| Blind, polar zone (beyond 84°N / 80°S) | 31.2 | 43.7 | 39.0 | +7.7 [−2.1, +17.5] |
| Blind, large-extent families | 70.9 | 31.8 | 67.3 | −3.6 [−34.0, +26.8] |
| Held-out, near transfer (US-ft, Albers, NaN nodata) | 56.5 | 60.3 | 60.7 | +4.3 [−5.5, +14.1] |
| Held-out, `us_ft` only | 56.7 | 64.0 | 67.5 | +10.9 [+1.5, +20.3] |
| Held-out, `nodata_nan` only | 41.5 | 43.5 | 42.6 | +1.1 [−14.3, +16.5] |
| Held-out, `csv_lon360` (adjacent) | 66.2 | 53.5 | 57.1 | −9.1 [−34.0, +15.9] |
| Held-out, `raster_southup` (far; E2 confirmatory) | 71.8 | 74.2 | 72.3 | +0.5 [−11.6, +12.6], uninformative |

- **Archive preservation adds little here:** `arch_coev − coev_tip_only` is +3.7 pp on blind positions (p = 0.44). The fixed mid-latitude `meta_T0` already reaches 63% in the UTM-valid zone (+26.5 over `naive`), while it collapses at the poles (25%) and on large extents (31%).
- **Why the blind gain (+22.4) is larger than the strict gain (+6.7).** Recomputed on the same records, it happens in three steps:
  1. Scoring per variant instead of all-or-nothing: +6.7 → +13.6.
  2. Dropping `low`, where `arch_coev` is slightly behind: +13.6 → +18.8.
  3. Fewer polar scenes in the blind set: +18.8 → +22.4.
- **The far-transfer comparison is uninformative by design.** Rasterio reads south-up rasters transparently, so ≥ 98.8% of `low`-correct scripts keep working under it for every method. Two other far-transfer candidates failed the pre-registered calibration rule for the same reason.

## Other results

**Weight internalization** (GSB v2, `train[20:60]`, LoRA on `gemma-4-26B-A4B-it`; [RESEARCH.md §2.5](docs/RESEARCH.md)):
- **The earlier `SFT-Weak` vs `SFT-Meta` gap is a teacher effect.** With the same `stage2_coev` teacher, weak filtering matches metamorphic filtering: Protocol A 60.2 vs 58.2%, Protocol B 50.2 vs 50.2%.
- **On bare-model data, the filter helps only on real OpenStreetMap.** On the 8-city set (96 tasks), `sft_base_meta` beats `sft_base_weak` by +16.7 pp [+2.1, +33.3], using a third of the data. On GSB v2 the two are within noise.
- **GRPO reward and training length:**
  - weak-reward and metamorphic-reward GRPO end within noise of each other;
  - 5 SFT epochs match or beat GRPO;
  - every model distilled from the strong teacher drops to 0% on `large_region_area_v2` and `large_route_length_v2`.

**Gate audit** ([RESEARCH.md §2.4](docs/RESEARCH.md)):
- On 1,384 unfiltered scripts, the full gate rejects 5.8–20.4% of oracle-correct scripts and accepts 0.8–8.5% of wrong ones.
- 15 of the 96 false rejections are gate errors:
  - a probe that changes the reference answer (6);
  - polar-probe polygons not split at the antimeridian (8);
  - a tolerance tighter than the oracle's (1).

  The probes for E1 fix the last two.
- Most other rejections are scripts that break on inputs the benchmark never tests: mixed CRSs, km or projected storage, polar scenes.
- On real OpenStreetMap, the relocating probes reject correct answers to absolute-position questions (northernmost-building latitude: 0/8 pass).

**Earlier, exploratory results** (single runs or few seeds, before the review; [RESEARCH.md §2–2.3](docs/RESEARCH.md)):
- **GSB v1, 4 seeds, `gemma-4-E4B-it`:** `metamorphic` 78.4 ± 3.8% vs `naive` 66.9 ± 22.7% vs DarwinX-style `preserve` 52.1 ± 22.4%; 2.1% lucky against 18.6% and 24.6%.
  - The built-in probes relocate to 58.3° / −56.2°, inside the `high` test band (52–64°). This is the overlap the blind protocol was designed to test on GSB v2.
  - With 4 seeds, `metamorphic − preserve` gives paired t p = 0.09.
  - The v0.2 runs included a short domain hint in the gate diagnostics, later removed. Without any diagnostic text, `meta_gate_only` reached 80.8 ± 2.0% (3 seeds).
- **GSB v1 polar stress test (86.5°N / 85.5°S):** the `metamorphic` harnesses converged to `estimate_utm_crs()` and fail loudly beyond the UTM range (83.7% errors, 2.7% silent wrong), rather than returning distorted numbers.
- **Real OpenStreetMap, 4 cities, 24 tasks:** `arch_coev` 86.1 ± 26.2% over 6 seeds (5/6 seeds at 91.7–100%), against 29.2% for `darwinx_soft`. Not re-run on the unified server.
- **`SciUnits` (non-GIS, 3 seeds) and `GeoAnalystBench` (17 tasks):** single small-sample results, reported in [RESEARCH.md §2.3](docs/RESEARCH.md).

## Next: E1, teacher-free best-of-K distillation

[docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md) fixes the design before any data:
- The bare model samples K = 16 candidate scripts per training task.
- One arm keeps a candidate that passes the weak check. The other keeps a candidate that also passes the fixed gate. The two arms draw with common random numbers.
- 6 replicate LoRA pairs, all trained from that one candidate pool.

The confirmatory test is `bok_meta − bok_weak` on Protocol A, an exact sign-flip test over the 6 replicates. A data-link test checks whether the gate-filtered pool is actually cleaner. The runner is `scripts/rebuttal/run_e1_bok.py` and the analysis `scripts/rebuttal/e1_summary.py`; both are committed.

## How the evidence was produced

- **Pre-registration.** Every analysis behind the status above was written down and committed before its data were read:
  - [docs/BLIND_PROTOCOL.md](docs/BLIND_PROTOCOL.md): the blind test set, the unified re-evaluation, and Addenda 1–4;
  - [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md): E1, E2, and Amendments 1–3.

  Every later change is a dated addendum, which states what had been seen when it was written.
- **Review.** [docs/REVIEW.md](docs/REVIEW.md) is a simulated top-venue review of an earlier state of the project (3/10). Several rounds of independent review followed: each point was checked against the raw records before it was adopted, and the corrections are recorded in RESEARCH.md.
- **Data.** `data/` and `runs/` are not in the repository. The benchmarks are rebuilt deterministically with the commands below, and the paths quoted in the docs refer to local artifacts.
- **Scoring environment.** Blind and held-out scores ran on hosts with identical library versions (Addendum 4). A cross-check of local against cloud scoring had found 38 disagreements in 26,290 runs.

## Benchmarks

**GSB v1** (`benchmarks/gsb`, `data/v1`):
- 21 families (18 seen, 3 held-out).
- Every task places the same local geometry at 3 locations: `low` (within ±7° of the equator, where Web Mercator passes by luck), `high` (52–64°) and `zone_edge` (on a UTM zone boundary).
- The test set has 480 tasks. Ground truth comes from a local AEQD projection, or `pyproj.Geod` for large extents.

**GSB v2** (`benchmarks/gsb_v2`, `data/v2`):
- 1,950 tasks, 20 families (15 seen, 5 held-out) in 4 groups:
  - multi-step vector pipelines;
  - raster/DEM + vector workflows (GeoTIFF with nodata);
  - large-extent geodesic tasks;
  - non-projection controls.
- Local tasks have `low`, `high`, `polar` (85–87°) and antimeridian variants. Raster tasks skip the antimeridian variant, and large-extent tasks use `low`, `mid` and `zone_edge`.
- The test set has 450 tasks (300 seen, 150 held-out).
- The audit is in [docs/gsb_v2_audit.log](docs/gsb_v2_audit.log): the reference passes 1,950/1,950, a UTM script 0/1,950, and a Web Mercator script 0/1,710 on the projection, raster and large-extent families.

**Blind relocation** (`benchmarks/gsb_v2/build_blind.py`, `data/v2_blind`):
- 8 positions per test task, 3,553 scored variants.
- Local families get 2 positions per |latitude| band (0–30°, 30–60°, 60–80°, 80–89°), area-uniform within the band. Large families are area-uniform over their valid range.

**Held-out mechanisms (E2)** (`benchmarks/gsb_v2/build_heldout_mech.py`, `data/v2_heldout_mech`):
- 1,705 scored variants. Each stores the `low` scene in a different convention:
  - near: US-survey-foot TM, local Albers, NaN nodata;
  - adjacent: CSV longitudes in [0, 360);
  - far: south-up GeoTIFF;
  - descriptive: Z = 0 vectors, multi-part geometries.

**Real OpenStreetMap:**
- `data/osm_eval`: 4 cities, 24 tasks.
- `data/osm_global`: 8 cities × 12 templates, 96 tasks.

**`SciUnits`** (`benchmarks/sci_units`): a non-GIS benchmark with 12 families of unit, calibration and missing-sentinel invariance.

**Scoring:**
- A real-valued answer is correct within 1% of the truth; counts must be exact.
- Categories:
  - `correct`: every variant right;
  - `lucky`: some right;
  - `silent_wrong`: runs everywhere, wrong everywhere;
  - `error`: exception, timeout or no numeric output.

## Repository layout

```text
meta-rsi/
├── meta_rsi/
│   ├── evolve.py               # evolution loop CLI
│   ├── meta_agent.py           # domain-neutral meta-agent
│   ├── archive.py              # lineage archive, parent selection, gating policies
│   ├── evaluator.py            # in-loop weak verifier + metamorphic gate vs hidden oracle
│   ├── metamorphic.py          # built-in invariance probes T(x)
│   ├── verifier_agent.py       # probe proposal and self-certification
│   ├── sandbox.py              # AST validation, harness loading, sandboxed execution
│   ├── llm.py                  # LLM backends (OpenAI-compatible / vLLM, Gemini, mock)
│   ├── harnesses/seed.py       # generation-0 harness
│   ├── training/               # trajectory collection, SFT/GRPO two-timescale training, LoRA sync
│   ├── eval/                   # test-set / polar evaluators, multi-seed and unified summaries, tables
│   └── infra/                  # Colab G4 vLLM launchers
├── benchmarks/
│   ├── gsb/                    # GSB v1
│   ├── gsb_v2/                 # GSB v2, blind relocation, held-out mechanisms, OSM builders
│   └── sci_units/              # SciUnits (non-GIS)
├── scripts/
│   ├── rebuttal/               # unified re-evaluation, blind/E2 scoring, gate audits, E1 runner and analysis
│   ├── wave_s4_2/              # GSB v2 weight-internalization runs
│   └── *.py                    # zero-shot baselines, error classification, figures
├── tests/                      # pytest suite
├── docs/
│   ├── RESEARCH.md             # experiment log, verified results, risk tracker, roadmap
│   ├── BLIND_PROTOCOL.md       # pre-registered blind / unified analysis and addenda
│   ├── NEXT_EXPERIMENTS.md     # pre-registered E1 / E2 and amendments
│   ├── REVIEW.md               # simulated top-venue review (Chinese)
│   ├── LITERATURE.md           # literature survey and positioning
│   ├── GSB_V1_BASELINE.md      # GSB v1 zero-shot baselines and error taxonomy
│   ├── INFRA_AND_GOTCHAS.md    # compute, budgets, Colab runbook
│   └── gsb_v{1,2}_audit.log    # benchmark separability audits
├── pyproject.toml
└── shell.nix                   # reproducible NixOS geospatial environment
```

## Quick start

Enter the environment (or install from `pyproject.toml`):

```bash
nix-shell
```

Build and audit GSB v2:

```bash
python -m benchmarks.gsb_v2.build --out data/v2 --train 100 --test 20 --heldout 30 --seed 0
```

```bash
python -m benchmarks.gsb_v2.audit data/v2
```

Build the blind and held-out-mechanism test sets:

```bash
python -m benchmarks.gsb_v2.build_blind
```

```bash
python -m benchmarks.gsb_v2.build_heldout_mech
```

Evolve a harness (GSB v2, metamorphic gate, the same vLLM-served model as task and meta model). The certified probes and the two-stage co-evolution are set up by the run scripts under `scripts/`:

```bash
python -m meta_rsi.evolve --exp my_run --data-root data/v2 --gate metamorphic --generations 6 \
    --task-backend openai-compat --task-url http://127.0.0.1:8000/v1 --task-model google/gemma-4-26B-A4B-it --task-no-think \
    --meta-backend openai-compat --meta-url http://127.0.0.1:8000/v1 --meta-model google/gemma-4-26B-A4B-it --meta-no-think
```

Evaluate a harness on the test split (Protocol A):

```bash
python -m meta_rsi.eval.eval_test --harness runs/my_run/gen_006/harness.py --data-root data/v2 --out runs/eval/my_run.json --task-no-think
```

Reproduce the unified, blind and held-out summaries from a finished `runs/v10_unified_6seed/`:

```bash
python -m meta_rsi.eval.unified_summary runs/v10_unified_6seed --ref naive
```

```bash
python scripts/rebuttal/score_blind.py summary
```

```bash
python scripts/rebuttal/score_blind.py --blind data/v2_heldout_mech summary
```

Run the tests:

```bash
python -m pytest -q
```

## Documentation

- [docs/RESEARCH.md](docs/RESEARCH.md): the experiment log with every verified result, the methodological risk tracker, and the roadmap.
- [docs/BLIND_PROTOCOL.md](docs/BLIND_PROTOCOL.md) and [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md): the pre-registered analyses.
- [docs/LITERATURE.md](docs/LITERATURE.md): related work on harness evolution, GeoAI benchmarks, silent errors and spurious correctness.
- [docs/GSB_V1_BASELINE.md](docs/GSB_V1_BASELINE.md): GSB v1 zero-shot baselines and the error taxonomy.
- [docs/INFRA_AND_GOTCHAS.md](docs/INFRA_AND_GOTCHAS.md): compute inventory and runbook.

## License

MIT, see [LICENSE](LICENSE).

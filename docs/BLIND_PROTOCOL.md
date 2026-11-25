# Blind Relocation Protocol for GSB v2 (pre-registered 2026-11-25)

Addresses review item M1 (`docs/REVIEW.md`): the gate's probes and the designed test variants sit in the same places. The built-in probes relocate data to 58.3N / 56.2S and the certified polar probe to 86N. The designed variants are `high` (54–64°), `polar` (85.3–87.2°) and `zone_edge` (UTM zone boundaries). A gain on those variants can therefore come from where the probes are placed rather than from the method itself.

This file fixes the blind test set and the analysis **before any method is scored on it**. Once scoring has started, the sampler and the analysis stay frozen. Any later change needs a new `--blind-seed` and a new section in this file, and results from both versions are reported.

## Test set (`benchmarks/gsb_v2/build_blind.py` -> `data/v2_blind/`)

- **Task content.** The 450 GSB v2 test tasks are used unchanged: same files, prompt, parameters and `low` workspace. The content is recovered by replaying the seeded `data/v2` build, and every replayed task (train and test) is checked against `data/v2/tasks.jsonl`. Blind locations use their own seed (`--blind-seed 20261125`).
- **Locations.** Each task gets 8 blind variants (3,600 in total), drawn without reference to any probe location:
  - *Local families (17).* 2 centres per |latitude| band [0,30), [30,60), [60,80), [80,89]. Within each band the latitude is uniform over the sphere's area (density ∝ cos lat), the hemisphere is random, and the longitude is uniform. Raster scenes are moved off the antimeridian because the generator cannot write GeoTIFFs that straddle it.
  - *Large families (3).* Latitude is area-uniform and longitude is uniform, over the whole range that keeps the scene within ±80° latitude and ±180° longitude.
- **Storage.** The benchmark's own conventions are used, keyed by latitude instead of by variant name:
  - GeoJSON and CSV in WGS84.
  - GeoPackages that a task stores projected go in UTM; beyond 84N / 80S they go in polar stereographic (EPSG:3413 / 3031); WGS84 if the scene crosses 180°.
  - GeoTIFFs follow the `low` convention (EPSG:3857, nodata −9999) within 2.2° of the equator, the `high` convention (EPSG:3857 or 4326, nodata −32768) up to 85°, and the `polar` convention beyond 85°.
- **Truth.** Each family's reference solver recomputes the truth on the files as written.
- **Exclusion rule.** A local variant is not scored (`scored: false`) if its truth differs from the `low` truth by more than build.py's own invariance tolerance (0.8%, or any difference for counts). The task content is the same, so a moving truth means the storage made the variant ill-posed. This excludes 47 of 3,600 variants (1.3%), all raster tasks stored in EPSG:3857 at 60–89°, where resampling shifts pixel-area answers by 1–2%, about the scoring tolerance.
- **Build check.** 0 non-finite truths; 2 vector scenes cross 180°. A script using EPSG:3857 fails almost every blind variant away from the equator, a UTM script fails most of the 80–89° band, and a local azimuthal-equidistant script passes every band.

What "blind" means here: the locations were drawn from a fixed distribution chosen without looking at the probes or at any method's results. They do not avoid the probe latitudes. The distance analysis below separates variants near the probe latitudes from those far from them.

## Scoring (`scripts/rebuttal/score_blind.py`)

- **Which scripts.** Every Protocol A eval in the unified 6-seed re-evaluation (`scripts/rebuttal/unified_6seed_eval.py`) saves the script each harness wrote per task on `low` (`<eval>.code.jsonl`). Those exact scripts are run offline on every scored blind variant with the oracle's runner (`benchmarks.gsb.harness.run_variant`, 25 s timeout) and judged by its tolerance (`matches`). Blind scoring makes no new LLM calls.
- **Primary outcome: blind accuracy.** For each task, the share of its scored blind variants answered correctly, averaged over the method's Protocol A repeats, then over tasks, then over seeds.
- **Primary comparisons.** Every method against `naive`, and every method against `arch_coev`. For each comparison:
  - seed-paired mean difference;
  - exact two-sided sign-flip test over seeds, Holm-adjusted across the methods compared with the same reference;
  - 95% hierarchical bootstrap interval, resampling seeds (paired) and task families.
- **Secondary breakdowns.** The same seed-paired difference is reported in each of these strata:
  - latitude band;
  - |latitude − nearest relocating-probe latitude| (58.3, −56.2, 86.0) in bins [0,5), [5,15), [15,90) degrees, local families only;
  - large families on their own;
  - `designed`: the same scripts on the designed non-low variants, taken from the Protocol A records.

## Pre-specified readings

1. **The gate generalizes beyond its probes.** A gated method's advantage over `naive` on blind variants is about as large as on the designed variants, and stays positive in the ≥15° probe-distance bin.
2. **The advantage is tied to probe placement (M1 holds).** The advantage concentrates in the [0,5) probe-distance bin and in the bands that contain probe latitudes, and is near zero (the 95% interval includes 0) in the ≥15° bin.
3. **Absolute blind accuracy is reported for every method and band, whichever reading holds.** A method that is wrong at 30–60° is wrong for most of the Earth's land, whatever the paired difference shows.

Whichever reading the data support goes into `docs/RESEARCH.md` with the full table.

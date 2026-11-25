# Next Experiments (pre-registered 2026-11-25, before any result)

Two experiments that answer the open points of the independent review (`runs/agent_mail/carbon_review_1.md`). Both are fixed here before data collection. Any change needs a dated section below, and results under both versions are reported. Decision rules follow `docs/BLIND_PROTOCOL.md` Addendum 2: one confirmatory comparison per experiment, everything else descriptive.

## E1. Teacher-free best-of-K self-distillation: does the metamorphic filter add value once the model can write candidates?

**Why.** §2.5 shows two things:
- With a strong teacher, the filter makes no difference.
- With bare-model data and one sample per task, the filter mostly shrinks the training set to 4 easy families (142/147 scripts), so it is never tested where it could matter.

Sampling K candidates per task removes both confounds: there is no evolved teacher, and the filter gets candidates to choose from.

**Data.**
- Bare `google/gemma-4-26B-A4B-it` with `harnesses/seed.py` on `train[20:60]` (600 tasks).
- K = 16 samples per task, temperature 0.8, `max_tokens` as in Wave S4-2.

**Gate.** The full gate used for `base_meta` (built-in probes + 3 certified probes), with the two fixes from §2.4:
- relocated geometries are split at the antimeridian;
- the agreement tolerance is 2%, twice the oracle's.

**Arms.**
- Let T be the set of tasks with at least one candidate passing both the weak check and the gate.
- For each task in T:
  - `bok_weak` takes one candidate drawn uniformly from those that pass the weak check (correct on `low`);
  - `bok_meta` takes one candidate drawn uniformly from those that also pass the gate.
- Both arms therefore have the same tasks and the same size. They differ only in which script is kept.
- 3 replicates, each with its own data-draw seed and training seed, so 6 LoRAs. SFT recipe as for `sft_meta` (2 epochs).

**Evaluation.**
- Protocol A on all 450 GSB v2 test tasks, temperature 0, 1 run per model.
- Protocol B on the same 100-task subset as the unified re-evaluation.
- 8-city OSM.

**Reported first.**
- |T| and its per-family composition.
- The number of tasks in T where the two arms keep different scripts.
- The oracle-correct share of each arm's training scripts.

**Confirmatory comparison.**
- Statistic: `bok_meta − bok_weak` on Protocol A correct, paired by replicate and task.
- Established only if both hold:
  - the 95% family-cluster bootstrap interval of the difference, pooled over the 3 replicates, excludes 0;
  - all 3 replicates agree in sign.
- With 3 replicates the sign-flip test cannot reach 0.05, so it is reported but not used.

**Descriptive.** Protocol B; OSM; each arm against the bare model and against `sft_base_meta` / `sft_base_weak`; silent and loud failures together.

**Pre-specified readings.**
1. **Filter adds value given candidates:** confirmatory difference established and positive.
2. **No added value:** the interval includes 0. The paper then drops the claim that metamorphic filtering improves training data beyond the weak check.
3. **Filter hurts:** established and negative, which would point to its false rejections (§2.4).

## E2. Held-out failure mechanisms (M1, CPU only)

**Why.** The blind relocation variants (`data/v2_blind`) test positions, within mechanisms the probes already exercise: Web Mercator distortion, UTM limits, projected storage in Mercator or transverse Mercator, km units. M1's broader point is that the probes and the test share mechanisms. E2 adds variants whose mechanism no probe uses, at the `low` location, so that only the mechanism changes.

**Variants** (per test task, from its `low` workspace; reference solvers recompute the truth; the exclusion rule of `data/v2_blind` applies):
- *Near transfer* (a mechanism family the probes touch, in a form they do not use):
  - `us_ft`: projected vector layers stored in a transverse Mercator CRS with `+units=us-ft`;
  - `albers`: projected vector layers stored in a local Albers equal-area conic, in metres.
- *Far transfer* (no probe touches the mechanism):
  - `nodata_nan`: rasters stored as float with NaN nodata instead of the integer sentinel;
  - `csv_lon360`: point CSVs with longitudes written in the 0–360° convention, west of Greenwich only, so the values actually change.

**Scoring.** The saved Protocol A scripts of the unified re-evaluation (`<eval>.code.jsonl`) are run offline with `run_variant` and `matches`, exactly as for the blind variants. No new LLM calls.

**Confirmatory comparison.** `arch_coev − naive` on far-transfer accuracy, seed-paired, established only if the exact sign-flip p < 0.05. Near transfer, each variant on its own, and every other method are descriptive.

**Readings.**
- If `arch_coev` (or `meta_gate_only`) beats `naive` on near transfer but not on far transfer, the gain is tied to the mechanisms the probes exercise. That supports M1.
- A gain on far transfer as well is evidence against M1's broader point.

## Order

1. E2 can be built and scored on CPU while the GPU finishes the unified re-evaluation.
2. E1 runs on `w3-rebuttal` after the unified re-evaluation. Its data collection must not share the server with those jobs.

## Amendment 1 (2026-11-25, 23:45 CST): after the reviewer's second round, before E1 data and before any E2 comparison

The reviewer's second round (`runs/agent_mail/carbon_reply_1.md`) raised three problems. I checked each one before amending.

**Disclosure.** When this was written, Antigravity's CPU scoring of the four original E2 mechanisms had started, beginning with `naive` and `arch_coev`. Its log prints each eval's accuracy pooled over all mechanisms. No accuracy by transfer class and no paired comparison had been computed. I (Claude Code) had seen no E2 number. The reasons below rest on the review and on the probe code, not on any result.

### E1: confirmatory rule replaced

- **Problem.** The rule as registered had two conditions: the pooled family-cluster bootstrap interval excludes 0, and 3/3 replicates agree in sign. The pooled bootstrap ignores replicate-to-replicate variance (data draw and LoRA training). Three null replicates agree in sign 25% of the time, and given that they agree, the bootstrap nearly always excludes 0.
- **Check.** Null simulation, 20 families, 3 replicates (`σ_rep` replicate variance, `σ_rf` replicate × family variance):

  | `σ_rep`, `σ_rf` (pp) | Registered rule | t₂ on replicate differences |
  |---|---|---|
  | 1.5, 3.0 | 20.4% | 4.6% |
  | 3.0, 4.0 | 24.8% | 5.9% |
  | 1.0, 4.0 | 13.4% | 5.1% |

- **New rule.** 6 replicates instead of 3, so 12 LoRAs.
  - Each replicate has its own data-draw seed and training seed.
  - All replicates draw from the one K = 16 candidate pool, so the inference is conditional on that pool.
  - Confirmatory: the exact two-sided sign-flip test on the 6 replicate-level differences `bok_meta − bok_weak` (Protocol A correct, each replicate pairing one `bok_meta` LoRA with one `bok_weak` LoRA). Established only if p < 0.05, which requires all 6 to agree in sign. This matches the 6-seed rule of Addendum 2.
  - The pooled family-cluster bootstrap and the t₅ interval are reported as descriptive.
- **Gate fix detail.** `probe_r2` relocates continental `large_*` scenes to 86°N, and some of them then contain the pole. Splitting such a polygon at ±180° alone leaves an open ring; the polar cap has to be closed along ±180° up to (±180, 90). Either close it, or skip polar relocation for scenes spanning ≥ 5°, as `metamorphic.py` does. In both cases, check that every relocated polygon keeps its geodesic area within 0.1% and that no single ring spans more than 180° of longitude.

### E2: transfer classes corrected, a far mechanism added

- **Problem A (unperturbed tasks).** Already satisfied by the build: a variant exists only for a task in which it changes at least one input file, and tasks without a variant are left out of that variant's accuracy. `us_ft` and `albers` reproject every `.gpkg` layer (256 tasks), not only the 70 with projected storage.
- **Problem B (`csv_lon360` is not far).**
  - Confirmed: `estimate_utm_crs()` raises "Unable to determine UTM CRS" for a point at lon 285°, the same error the polar probe triggers at 86°N.
  - PROJ `aeqd` wraps the longitude (lon 285° centred on −75° maps to (0, 0)).
  - So a script that learned the AEQD fallback from the polar probe passes `csv_lon360` without any longitude handling.
- **A further problem, found while checking B.** `nodata_nan` is not far either. The built-in `crs_storage_flip` probe already swaps the nodata sentinel (−9999 ↔ −32768) and stores float32 (`metamorphic.py`, `flip_crs_and_nodata`), so nodata handling is a mechanism the gate exercises. The NaN form is new, which makes it near transfer.
- **New classes.**
  - Near: `us_ft`, `albers`, `nodata_nan`.
  - Adjacent: `csv_lon360` (no probe writes it, but the polar probe's fix handles it); descriptive only.
  - Far: `raster_southup`, new.
    - Every GeoTIFF is stored south-up: rows flipped, the transform's `e` made positive, same pixels and footprint, no resampling.
    - No built-in or certified probe writes it. All of them keep the source raster's row order (`metamorphic.py`, `probes_e1/probe_r2_polar_laea.py` keep the sign of `e`).
    - Built for all 160 raster tasks (7 families). Reference truths are unchanged in all 160 (relative difference 0), so all are scored. Rasterio reports such a raster's bounds with bottom > top, which a script that trusts `bounds` or hand-rolls row/column arithmetic will get wrong.
- **Confirmatory comparison (form unchanged).** `arch_coev − naive` on far-transfer accuracy, now `raster_southup` only, seed-paired, exact sign-flip p < 0.05.
- **Added descriptive outputs.**
  - `drop`: for the same script, variant correct minus `low` correct, per task. Variant accuracy mixes competence on `low` with robustness to the mechanism, and `drop` separates the two.
  - `lowok`: accuracy only on tasks the script solved on `low`.
  - `matched`: near against far on the tasks that have both, so that the two are not compared across different families.
- **Added readings.**
  - A far-transfer difference counts as transfer of robustness only if `arch_coev − naive` on `drop` (far) has the same sign. Otherwise it is reported as a difference in competence on `low`.
  - If both methods keep ≥ 90% `lowok` accuracy on `raster_southup`, the mechanism breaks too few scripts to separate them, and E2's far result is reported as uninformative about M1.

## Amendment 2 (2026-11-26, 00:20 CST): after the reviewer's third round, before E1 data and before any E2 comparison

Source: `runs/agent_mail/carbon_reply_2.md`. Each point was checked first. Nothing from the unified re-evaluation was read.

### E2: `raster_southup` is uninformative; a calibration rule for its replacement

- **Finding.** Rasterio and GDAL read a south-up raster through the same idioms as a north-up one: `mask`, `geometry_mask`, `calculate_default_transform` + `reproject`, `res`, and `np.gradient` all give identical results. Saved test-set scripts of Stage 3 models (`runs/v9_rebuttal_gpu`) that were correct on `low` keep 100% on `raster_southup`: 91/91 for `base_26b` rep 1 and 100/100 for `sft_meta` rep 1 (my check, `scripts/rebuttal/calibrate_heldout_mech.py`). The reviewer found the same on 643 scripts. Under Amendment 1's rule it is uninformative, so it stays as a descriptive mechanism only.
- **Candidates for far transfer.** Neither is written by any built-in or certified probe; `metamorphic.py` and the certified probes never add Z or promote geometry types.
  - `vector_z`: every vector layer in `.gpkg` or `.geojson` written with a Z coordinate of 0. Breaks code that unpacks `x, y` from coordinates.
  - `multipart`: every geometry in `.gpkg` or `.geojson` written as its multi-part type, one part each. Breaks code that reads `.exterior`, `.x` or `.coords` from a single-part geometry.
- **Entry rule** (fixed before running the calibration). A candidate enters far transfer only if all three hold:
  1. no probe writes it;
  2. reference truths are unchanged under the invariance rule;
  3. among scripts correct on `low`, its retention lies in [10%, 90%]. Retention is pooled over the two calibration sources, `base_26b` rep 1 and `sft_meta` rep 1 in `runs/v9_rebuttal_gpu`. These are Stage 3 models outside the unified comparison.

  Every candidate that qualifies enters far transfer; each task's far accuracy averages its far variants. If none qualifies, far transfer is `raster_southup` alone, and the confirmatory comparison will be reported as uninformative.
- **Readings (replace Amendment 1's `drop` reading).**
  - `drop` (variant correct − `low` correct) penalizes a method for higher `low` accuracy: tasks already wrong on `low` contribute 0, so a more accurate and more robust method can show the larger drop. It is removed.
  - Replacement: `bothlow`. For each seed, only tasks that both methods solve on `low` in every repeat of that seed are kept; the methods are compared on the variants of those tasks, and an exact sign-flip test runs over seeds.
  - A far-transfer difference counts as transfer of robustness only if it is established and the `bothlow` difference has the same sign.
  - `lowok` stays in the per-method table but is no longer compared in pairs. Requiring all 12 runs to solve `low` dropped hard tasks.
  - Uninformative: both methods keep ≥ 90% `lowok` on far transfer (ceiling), or both keep ≤ 10% (floor).

### E1: scope of the replicate test, draws, and the gate fix as built

- **Shared candidate pool.** All 6 replicates draw from one K = 16 pool. That pool's luck is shared across replicates, so under a null that also resamples the pool, the replicate differences are positively correlated. The reviewer's simulation puts the 6/6 sign-flip error at 6.6–14.1% for ρ = 0.10–0.27.
  - The confirmatory claim is therefore stated **conditional on the realized pool**: filtering this pool changes the trained model beyond tie-breaking and training noise.
  - **Added, pre-specified.** The link from gate to cleaner data is tested separately. For each task in T, compute the oracle-correct share of `pool_m` minus that of `pool_w`, average within each family, and run an exact sign-flip over the 15 families.
  - A positive confirmatory result is attributed to cleaner data only if this difference is positive with p < 0.05.
  - Independent pools per replicate were considered and rejected for cost (3× the sampling at K = 8).
- **Draws with common random numbers.** Per replicate and task, one random permutation of the candidates. `bok_weak` takes the first candidate that passes the weak check; `bok_meta` takes the first that also passes the gate. Both marginals stay uniform over their pools. The arms keep different scripts only when the weak pick fails the gate, so the count of differing scripts is the number of tasks where the filter acted.
- **Bare model.** Evaluated on the same server, alongside the 12 LoRAs.
- **`probe_r2` as built.** Geometries spanning > 0.5° in geographic coordinates are densified along the geodesic every 10 km before relocation. Polygons are split at ±180°; those that enclose the pole are split along the 0°/180° diameter, with a 25 m polar cap removed.
  - Densifying is needed for the split to preserve area: without it the error is 0.1–1.3%; with it, at most 0.003% over 207 polygons.
  - Side effect: the relocated edges no longer expose vertex-only reprojection, so the 6 polar rejections of well-formed polygons and 3 polar-curvature `other` rejections also clear.
- **Regression of both fixes** (my rerun of the full gate on 1,384 scripts; `runs/v11_gate_precision/gate_vs_oracle_*_e1.jsonl`):
  - Of the 96 false rejections, 18 clear and 78 stay; no script newly fails.
  - 6 oracle-wrong scripts newly pass. One was rejected only by the unsplit-polygon bug. The other five drift 1.1–2.0% (tolerance): false acceptance rises 0.8→1.9%, 8.5→10.6% and 1.3→1.9% in the three populations.
  - The tolerance is settled in Amendment 3, after a rerun with the fixed probes at 1% separates the two fixes.

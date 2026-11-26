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

## Addendum: unified 6-seed re-evaluation (pre-registered 2026-11-25, 22:05 CST)

Written while `runs/v10_unified_6seed/` was at job 9 of 144. No unified result had been read when this was written. The only blind numbers seen were four per-method accuracies for `meta_gate_only`, which compare nothing.

**What is run.** All 48 best harnesses (8 methods × 6 seeds) on one server: Colab G4 vLLM bf16, `google/gemma-4-26B-A4B-it`, temperature 0, no thinking, `--max-model-len 8192`.
- Protocol A: 2 repeats on all 450 test tasks. For 15 harnesses (seeds 1–5 of `naive`, `meta_gate_only`, `oracle_soft`), repeat 1 reuses the rebuttal-session eval: same model and decoding, but that server also had `--enable-lora`.
- Protocol B: 1 repeat on the first 5 test tasks of each family (100 tasks).
- Tooling: `scripts/rebuttal/unified_6seed_eval.py launch --a-reps 2 --b-reps 1 --b-per-family 5`.

**Primary outcome.** The Protocol A share of tasks correct on every variant, averaged over the 2 repeats, per seed.

**Primary comparisons** (`python -m meta_rsi.eval.unified_summary`, run with `--ref naive` and with `--ref arch_coev`). For each method against each reference:
- seed-paired mean difference;
- 95% hierarchical bootstrap interval, resampling seeds (paired) and task families;
- exact sign-flip p over seeds, Holm-adjusted across the 7 other methods.

With 6 seeds the smallest sign-flip p is 0.031, so no Holm-adjusted p can fall below 0.05. The decision rule is therefore the bootstrap interval: a difference counts as established only if the interval excludes 0. The sign-flip p is reported next to it.

**The seed-0 question.** The original evals put seed 0 far above seeds 1–5 for `naive`, `meta_gate_only` and `oracle_soft` (46.9 / 53.6 / 55.3% against 31.9 / 33.9 / 36.0%). For every method we report the per-seed shift from the original eval, and the gap between seed 0 and the mean of seeds 1–5, under both the original and the unified evals.
- (a) If the unified seed-0 gap for those three methods is within repeat noise (≤ 4 pp), the original seed-0 numbers reflect eval drift, and the single-seed comparisons in RESEARCH.md §2.3.1 are withdrawn.
- (b) If the gap stays above 10 pp, seed 0 is a genuinely favourable seed. The single-seed comparisons are then reported as one seed out of six.
- Anything in between is reported as is.

**Backend check.** `arch_coev` was first evaluated on Vertex MaaS. Its shift from that eval is reported per seed. For the 15 reused harnesses, |rep 1 − rep 2| is compared with the same quantity for the 33 harnesses whose both repeats ran on the unified server. If the reused gap is larger by more than 2 pp on average, the reused repeat 1 is dropped and those harnesses use repeat 2 only.

**Secondary.**
- Protocol B on its 100-task subset, with the same comparisons.
- What Protocol B does to tasks that Protocol A scored correct, lucky or silent-wrong.
- Protocol A repeat variability.
- The blind-variant analysis specified above.

**Reporting.** The unified table replaces the §2.3.1 multi-seed table as the main 6-seed result in `docs/RESEARCH.md`, and the original numbers stay alongside it. Whichever reading holds is reported.

## Addendum 2: amendments after independent review (2026-11-25, 23:10 CST, before any unified analysis)

An independent reviewer (Antigravity, `gemini-4-carbon-high`, read-only; `runs/agent_mail/carbon_review_1.md`) found flaws in the two sections above. The amendments below were fixed before any unified or blind comparison was computed.

Disclosure of what was seen first:
- 26 of 48 `A_rep1` evals and 15 `.blind.json` files (seeds 1–5 of `meta_gate_only`, `naive`, `oracle_soft`, written 21:35–22:31) were on disk.
- The reviewer ran `score_blind.summarize` on those 15 files and quoted one result in its review: `oracle_soft − naive` in the ≥15° bin, interval [+1.3, +21.4].
- While checking run status I (Claude Code, who writes the analysis) saw one log line with the held-out accuracy of one unidentified harness. No other unified or blind number was read.
- Every amendment either makes the decision rule stricter or adds an analysis. The original analyses are still reported.

**1. Decision rule (replaces the bootstrap rule of Addendum 1).**
- With 6 seed clusters the percentile bootstrap is too narrow. Its standard error is biased by a factor √(5/6), and it uses normal rather than t₅ quantiles. Under the null, "interval excludes 0" fires about 13% of the time instead of 5%, and far more often across 14 comparisons.
- There is now **one confirmatory comparison**: `arch_coev − naive` on the primary outcome. It is established only if the exact two-sided sign-flip test over the 6 seeds gives p < 0.05, which requires all 6 seeds to agree in sign. It is reported with a 95% t₅ interval on the 6 seed-level differences.
- **All other comparisons are descriptive**: every method against `naive` and against `arch_coev`, under both protocols, and for blind accuracy. They are reported with Bonferroni-simultaneous t₅ intervals over the 14 primary-outcome comparisons (per-comparison level 1 − 0.05/14), together with unadjusted and Holm-adjusted sign-flip p. The seeds × families bootstrap is kept as a descriptive interval only. No descriptive comparison is called significant or established.
- Seeds are not added after seeing these results. The earlier suggestion of seeds 6–9 is withdrawn.

**2. Search budget is reported with every 6-seed result.**
- `arch_coev` and `coev_tip_only` run 8 Stage-2 generations on top of a 6-generation Stage-1 archive (14 candidate generations). Every other method runs 6.
- The 3 certified probes used by `meta_gate_only`, `coev_tip_only` and `arch_coev` (and by the Wave S4-2 filter and GRPO reward) were certified once, by `gemini-4-hydrogen` (`runs/v6_gsb_v2_s3b/verifier_r*_gem4h.json`), and reused for every seed.
- Test-time calls per task are reported alongside.

**3. Seed-0 readings (replaces Addendum 1).**
- The statistic is the mean, over `naive`, `meta_gate_only` and `oracle_soft`, of the unified seed-0 gap. Each method's gap is also shown.
- Mean ≤ 4 pp: the original seed-0 numbers reflect eval drift.
- Mean > 10 pp: the seed-0 advantage lives in the evolved harness itself, not in the eval server. A re-evaluation cannot tell whether that comes from the seed-0 dev split or from search luck, and no claim is made either way.
- In between: reported as is.
- For `coev_tip_only` and `arch_coev` the seed-0 gap is not interpreted, since their probes were certified once and reused.

**4. Backend check (the drop rule is unchanged, the comparison is tightened).**
- Report the signed mean of rep 1 − rep 2 and the mean of |rep 1 − rep 2|.
- Compare reused harnesses with fresh harnesses of single-call methods only: seed 0 of `naive`, `meta_gate_only` and `oracle_soft`, plus all `darwinx_soft` and `meta_T0` harnesses.
- Repeat 1 of the 15 reused harnesses is dropped from every analysis (Protocol A, A→B, blind) if either the mean |gap| exceeds the fresh single-call mean by more than 2 pp, or |signed mean| exceeds 2 pp.

**5. Added checks.**
- Protocol A accuracy on the 100-task Protocol B subset against all 450 tasks, per method.
- Silent failures (`lucky` + `silent_wrong`) and loud failures (`error`) are always reported together. A lower silent-failure rate is never reported as a win without the matching error rate. The primary endpoint stays `correct`.

**6. Blind variants: added strata and a narrower reading.**
- The original strata stay as pre-registered. The signed probe-latitude distance puts every 80–89°S variant in the ≥15° bin, although the polar probe at 86°N exercises the same mechanism there: UTM is undefined beyond 84°N / 80°S.
- Added stratum: distance in |latitude| to the probes' |latitude| (58.3, 56.2, 86.0), with the same bins.
- Added stratum: mechanism domains.
  - equatorial weak-proxy zone (|lat| ≤ 8°);
  - UTM-valid zone (8° < |lat|, lat ≤ 84°N, lat ≥ 80°S);
  - polar zone beyond the UTM limits;
  - antimeridian-crossing scenes, listed but not analysed (n = 2).
- Added: blind accuracy weighted by cos(latitude), to undo the 25% share given to the 80–89° band, which covers 1.5% of the sphere.
- **Reading 1 is narrowed.** Blind relocation tests whether a method generalizes across positions within the failure mechanisms the probes exercise (Web Mercator distortion away from the equator, UTM domain limits). It cannot test generalization to mechanisms the probes do not exercise. Reading 1 therefore becomes "the gain is not tied to the probes' coordinates". A positive result is not evidence against M1's broader point that the probes and the designed variants share failure mechanisms; that needs held-out mechanisms, which `data/v2_blind` does not contain.

## Addendum 3: area weight corrected (2026-11-25, 23:40 CST, before any blind summary)

The reviewer's second round (`runs/agent_mail/carbon_reply_1.md`) found that the cos(latitude) weight of Addendum 2 double-counts. `build_blind.py` already draws each centre area-uniformly within its band, so the only bias is the equal quota of 2 centres per band. Weighting by cos(latitude) on top gives each band a weight ∝ cos²φ / Δsin, which still over-weights 80–89° by about 2.2×.

- **Corrected weight.** For a local task, each scored variant in band b gets (sin hi_b − sin lo_b) / n_{t,b}. That is the band's share of the sphere's area (0–30°: 0.500, 30–60°: 0.366, 60–80°: 0.119, 80–89°: 0.015), split over the task's scored variants in that band. `large` tasks are sampled area-uniformly over one interval and keep equal weights.
- **Name.** The stratum is renamed `all_areaw`. `all_cosw` is no longer reported.
- **A related fix.** The per-task share divided by max(weight sum, 1). That is harmless for counts but wrong for fractional weights. It now divides by the weight sum itself.
- No blind summary had been computed when this was fixed, so no reported number changes.

## Addendum 4: one scoring environment and a timeout retry (2026-11-26, 12:55 CST, before any summary)

All 144 unified evals are in (48 × `A_rep1`, `A_rep2`, `B_rep1`, all clean). No unified, blind or held-out summary has been computed. The only numbers read are the per-record agreement counts below, which compare environments, not methods.

**Why.** The blind (`.blind.json`) and held-out (`.heldout.json`) sidecars come from running each harness's saved code on the variant workspaces in a sandbox, so they depend on the scoring machine. Local scoring stopped at 73/96 and 68/96 because the workstation was overloaded, and the rest was to run on Colab CPU hosts. Most of what was left was `arch_coev`'s second repeat. Mixing environments would then partly confound environment with method, and the E2 confirmatory comparison is `arch_coev − naive`.

**Cross-check.** Five files already scored locally were re-scored on a Colab host: `naive_s0_A_rep1`, `arch_coev_s0_A_rep1`, `naive_s1_A_rep2`, `oracle_soft_s1_A_rep2` and `meta_gate_only_s2_A_rep1`. Data were identical: both `tasks.jsonl` files and both variant trees have the same sha256 (`runs/agent_mail/data_hashes.txt`). The `correct` field disagreed on 38 of 26,290 (task, variant) pairs (0.14%; `runs/agent_mail/crosscheck_report.json`):
- 31: local `exception`, Colab `ok` and correct. These are 2 tasks of `meta_gate_only_s2` and 1 of `arch_coev_s0`.
- 6: local `ok` and correct, Colab `timeout` (1 task of `arch_coev_s0`, 25 s limit).
- 1: a numeric difference across the tolerance (0.4972 vs 0.5025).
- The other three files agreed exactly.

Library versions differ:

| Environment | Python | rasterio | GDAL | pyproj | PROJ | geopandas | numpy |
|---|---|---|---|---|---|---|---|
| Local NixOS | 3.14 | 1.5.1 | 3.13.3 | 3.8.0 | 9.9.0 | 1.2.0 | 2.5.4 |
| Colab | 3.13.18 | 1.5.2 | 3.12.2 | 3.8.0 | 9.8.1 | 1.2.0 | 2.4.6 |

The Colab host also has shapely 2.2.0 and scipy 1.16.3.

**Rule.**
1. **One environment.** All 96 `.blind.json` and 96 `.heldout.json` files are re-scored from scratch on Colab hosts with the library versions above. If the work is split across hosts, each host's versions and data hashes are checked first. The local sidecars are archived and enter no analysis.
2. **Timeout retry.** Wall-clock limits depend on host speed and load, which is why the 6 Colab timeouts above appeared. After the first pass, every (task, variant) whose status is `timeout` is run exactly once more, with `score_blind.py score --retry-timeouts` at no more than 4 workers and the same 25 s limit. The second result stands even if it times out again, and the record is marked `retried`. This applies to every method alike. The number of timeouts before and after the retry is reported per method.
3. **Main evals unchanged.** The Protocol A/B scores in the main eval files came from the eval server itself and are not re-scored.

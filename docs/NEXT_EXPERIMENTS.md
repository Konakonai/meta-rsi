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

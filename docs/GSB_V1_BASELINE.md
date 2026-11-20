# `benchmarks/gsb` v1 — Benchmark Audit & Zero-Shot Empirical Baseline

> Consolidated empirical foundation from `results/zero_shot_v1/` (collected 2026-11-18 to 2026-11-19 on Colab G4 + vLLM).
> All aggregate figures correspond directly to local JSON/JSONL artifacts in `results/zero_shot_v1/` and [docs/gsb_v1_audit.log](gsb_v1_audit.log).

---

## 1. Benchmark Design & Verification (`data/v1`)

- **Scale**: 3,180 tasks across 21 families (`18 seen × 150 train` = 2,700 train; `18 seen × 20 test + 3 held-out × 40 test` = 480 test).
- **Three task groups**:
  1. **Local projection-dependent (12 seen + 3 held-out)**: buffers, distances, areas, perimeters, overlays, road lengths (`density_in_polygon`, `road_buffer_area`, `max_pair_distance` held out).
  2. **Methodological traps (4 seen)**: 3 purely non-projection families (`points_in_polygon`, `area_weighted_population`, `mean_per_district`, where `EPSG:3857` passes `170/170`) plus `covered_population` (metric-buffer deduplication trap; projection-dependent, failing `EPSG:3857` on `166/170` tasks). `roads_in_district` additionally distinguishes `within`, `intersects`, and clipping.
  3. **Large-extent geodesic (2 seen)**: `large_region_area`, `large_route_length` spanning tens of degrees; ground truth via `pyproj.Geod` on WGS84.
- **Multi-location oracle (`low`, `high`/`mid`, `zone_edge`)**: Each task places identical local geometry at 3 global locations. Prompts omit location and CRS; 25% of `.gpkg` layers are stored in local UTM; point layers may be CSV (`lon`, `lat`).
- **Ground-truth accuracy check** (`scripts/check_aeqd_truth.py`, 1,000 random 20–30 km shapes across latitudes `0–64°`): bbox-centered AEQD agrees with ellipsoidal `pyproj.Geod` within **0.00007%** (area), **0.00003%** (length), and **0.00005%** (distance).

### Method-Separability Audit ([docs/gsb_v1_audit.log](gsb_v1_audit.log))

| Solver Method | Outcome on 3,180 Tasks (`data/v1`) |
|---|---|
| Reference (AEQD local / `Geod` large) | **3,180 / 3,180** correct |
| `estimate_utm_crs()` | All local tasks pass; fails large-extent (`158/170` area [`12/170` pass], `156/170` route [`14/170` pass]) |
| `EPSG:3857` (3-location oracle) | Fails 100% of length/area families; fails `147–166 / 170` count-by-distance tasks; passes non-projection families (by design) |
| `EPSG:3857` (**`low` variant only**) | **Spuriously passes** `147–164 / 170` length tasks and `73–79 / 170` area tasks |
| Fixed `EPSG:32630` (UTM 30N) | Fails `122–170 / 170` projection-dependent tasks |
| Spherical haversine (`large_route_length`) | Passes all 510 variants (max error `0.41% < 1%`) |

### Key Construction Lessons & Known v1 Limitations
- **Buffer overlap constraint**: When buffers do not overlap, Web Mercator distortion scales buffer radius and polygon area identically ($n \pi r^2$), cancelling out. `buffer_union_area` therefore constrains overlap ratio to `30%–80%`.
- **Large-route span**: Initial 25–45° spans caught UTM only 38% of the time; widening to 45–75° increased UTM catch rate to 92%.
- **`zone_edge` local sensitivity**: For <20 km local extents straddling a UTM boundary, projecting into either adjacent UTM zone introduces <0.1% length distortion (well below 1% tolerance). Thus `zone_edge` primarily catches hardcoded UTM zones rather than `estimate_utm_crs()`. Breaking `estimate_utm_crs()` requires **polar latitudes (`>84°N / <80°S`)**, now supported in `meta_rsi.eval.eval_ood` and planned for GSB v2.

---

## 2. Zero-Shot Evaluation on 480 Test Tasks (`× 8` Samples = 3,840 Answers / Model)

All five conditions ran with `scripts/zero_shot.py` on Colab G4 (`vLLM`, thinking ON, `Qwen3.8` effort=`low`, `max_tokens=6144`):

| Model / Prompt Condition | `pass@1` | `pass@8` | `Lucky` | `Silent Wrong` | `Error` | Proj. Errors | Median Tokens | Gen Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `e4b_think` (Gemma 4 E4B, base) | 22.4% | 39.6% | 26.9% | 29.3% | 21.4% | 50.4% | 1,406 | 444 s |
| `e4b_think_hint` (+1% accuracy requirement) | 24.2% | 52.3% | 22.0% | 27.8% | 26.0% | 42.8% | 1,660 | 537 s |
| `e4b_think_rules` (+1% + 4 explicit CRS rules) | 44.0% | 86.9% | 0.3% | 13.9% | **41.8%** | 2.0% | 2,167 | 875 s |
| `q27_low` (Qwen3.8-27B, effort=`low`) | 62.2% | 97.3% | 8.9% | 11.0% | 17.8% | 19.2% | 1,160 | 3,071 s |
| `g26b_think` (Gemma 4 26B-A4B) | 71.7% | 98.3% | 13.4% | 10.4% | 4.5% | 23.7% | 2,028 | 2,967 s |

### Single-Location Evaluation Inflation

Evaluating only the equatorial (`low`) variant dramatically overstates accuracy compared to the 3-location oracle:

| Run | 3-Location Oracle | `low` Only | Inflation (`low − Oracle`) | `high/mid` Only | `zone_edge` Only | Runnable on All 3 | Total Errors | `KeyError: 'id'` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `e4b_think` | 22.4% | 48.0% | **+25.6%** | 24.1% | 24.5% | 78.6% | 821 | 73 |
| `e4b_think_hint` | 24.2% | 45.1% | **+20.9%** | 25.7% | 26.5% | 74.0% | 997 | 67 |
| `g26b_think` | 71.7% | 84.9% | **+13.2%** | 72.3% | 72.5% | 95.5% | 172 | **99** |
| `q27_low` | 62.2% | 70.7% | **+8.5%** | 63.7% | 63.6% | 82.2% | 683 | 10 |

- Notice that **57.6% (99/172)** of Gemma 4 26B-A4B's runtime crashes are `KeyError: 'id'` after `sjoin`/`sjoin_nearest` renames duplicate `id` columns to `id_left`/`id_right` (concentrated in `covered_population` [54] and `count_near_roads` [44]).

### Per-Family Zero-Shot `pass@1 / pass@8`

| Family | `e4b_think` | `e4b_think_hint` | `g26b_think` | `q27_low` |
|---|---:|---:|---:|---:|
| `area_by_class` | 0.075 / 0.10 | 0.100 / 0.30 | 0.694 / 1.00 | 0.613 / 1.00 |
| `area_weighted_population` | 0.438 / 1.00 | 0.338 / 0.95 | 0.988 / 1.00 | 0.769 / 1.00 |
| `buffer_union_area` | 0.044 / 0.10 | 0.094 / 0.40 | 0.750 / 1.00 | 0.594 / 1.00 |
| `centroid_distance` | 0.100 / 0.40 | 0.200 / 0.80 | 0.794 / 1.00 | 0.719 / 1.00 |
| `count_near_roads` | 0.075 / 0.25 | 0.138 / 0.60 | 0.412 / 0.90 | 0.394 / 0.90 |
| `count_within` | 0.425 / 0.60 | 0.444 / 0.80 | 0.856 / 1.00 | 0.887 / 1.00 |
| `covered_population` | 0.013 / 0.10 | 0.056 / 0.20 | 0.300 / 0.85 | 0.531 / 1.00 |
| `density_in_polygon` (held-out) | 0.013 / 0.05 | 0.016 / 0.05 | 0.581 / 1.00 | 0.444 / 0.975 |
| `large_region_area` | 0.000 / 0.00 | 0.019 / 0.15 | 0.525 / 1.00 | 0.625 / 1.00 |
| `large_route_length` | 0.812 / 1.00 | 0.812 / 1.00 | 0.887 / 1.00 | 0.912 / 1.00 |
| `max_pair_distance` (held-out) | 0.428 / 0.775 | 0.500 / 0.975 | 0.887 / 1.00 | 0.816 / 1.00 |
| `mean_nn_distance` | 0.356 / 0.70 | 0.350 / 0.90 | 0.863 / 1.00 | 0.650 / 1.00 |
| `mean_per_district` | 0.225 / 0.75 | 0.269 / 0.85 | 0.875 / 1.00 | 0.619 / 0.80 |
| `nearest_facility_max` | 0.138 / 0.35 | 0.181 / 0.60 | 0.669 / 1.00 | 0.412 / 0.90 |
| `overlay_area` | 0.025 / 0.15 | 0.037 / 0.20 | 0.600 / 0.95 | 0.556 / 1.00 |
| `perimeter_by_class` | 0.175 / 0.25 | 0.156 / 0.35 | 0.688 / 1.00 | 0.606 / 1.00 |
| `points_in_polygon` | 0.787 / 0.95 | 0.769 / 0.90 | 1.000 / 1.00 | 0.894 / 1.00 |
| `road_buffer_area` (held-out) | 0.056 / 0.075 | 0.066 / 0.20 | 0.647 / 0.975 | 0.487 / 0.95 |
| `road_length_by_type` | 0.144 / 0.20 | 0.094 / 0.20 | 0.688 / 1.00 | 0.419 / 0.95 |
| `roads_in_district` | 0.069 / 0.10 | 0.087 / 0.20 | 0.537 / 0.95 | 0.406 / 0.95 |
| `sum_within` | 0.481 / 0.70 | 0.506 / 0.70 | 0.850 / 1.00 | 0.838 / 1.00 |

---

## 3. Silent-Error Attribution & Skill Contamination Findings

### 3.1 Rule + `gemini-decision-1` Error Taxonomy (5,754 Runnable-but-Wrong Answers)

- **Method**: Deterministic re-execution fingerprinting (`scripts/classify_errors.py`, pass r2) attributed **5,108 / 5,754 (88.8%)** runnable-but-wrong answers across the 4 base/hint/26B/27B runs; the 646 residual unattributed answers plus 85 newly added pass-2 cases (`latlon_order` + `nn_in_degrees`, 731 total) were classified by `gemini-decision-1` (`scripts/classify_llm.py`, v2 14-class schema).
- **Agreement & Calibration**:
  - Rules vs `gemini-decision-1` (5,012 rule-labeled answers, pass 1): **97.2% agreement**, Cohen's $\kappa = 0.948$ ($\kappa = 0.970$ for confidence $\ge 0.8$).
  - Independent blind spot-check on 60 random unattributed answers (v2 schema, Claude blind labeling): overall **82% agreement** ($\kappa = 0.79$); confidence $\ge 0.7$: **87%** ($\kappa = 0.85$); confidence $\ge 0.9$: **93%** ($\kappa = 0.92$).

Share of all 3,840 answers per run (v2 combined labels):

| Root Cause | `e4b_think` | `e4b_think_hint` | `g26b_think` | `q27_low` |
|---|---:|---:|---:|---:|
| `EPSG:3857` (Web Mercator) | 32.4% | 27.5% | 16.6% | 9.3% |
| Unprojected degrees | 11.4% | 9.1% | 5.1% | 4.9% |
| Hardcoded / wrong UTM zone | 6.8% | 5.9% | 0.0% | 0.7% |
| Cross-layer CRS mismatch | 2.9% | 2.8% | 0.8% | 2.4% |
| General programming bug (index alignment, azimuth-as-distance, etc.) | 0.7% | 1.7% | 0.0% | 0.8% |
| Single UTM zone on large-extent data | 0.0% | 0.1% | 1.2% | 0.6% |
| Swapped `(lon, lat)` arguments (e.g., `geopy.geodesic`) | 0.7% | 1.1% | 0.0% | 0.2% |
| Nearest-neighbor search in degree space + geodesic distance | 0.3% | 0.8% | 0.0% | 0.3% |
| Other (units, aggregation trap, predicate, UTM-as-deg) | 0.9% | 0.9% | 0.1% | 0.9% |
| **Total runnable-but-wrong (`lucky + silent_wrong`)** | **56.2%** | **49.8%** | **23.8%** | **20.0%** |

### 3.2 "Know-but-Don't-Use" CoT Evidence (Motivation for RSI / Execution Verification)

Models often explicitly verbalize Web Mercator distortion in their reasoning traces yet still emit `EPSG:3857` as a default shortcut (observed in `gis-grpo-pilot` logs, 2026-11-18):
- **Gemma 4 E4B**: *"for a general solution where the extent is unknown, 3857 is often the pragmatic choice in coding challenges unless high precision is strictly required."*
- **Qwen3.8-27B (`effort=low`)**: *"EPSG:3857 (Web Mercator) is common but distorts distances at high latitudes. A better approach might be to use a UTM zone or just use EPSG:3857. But the simplest and most common approach for this kind of task is to reproject to EPSG:3857"*

### 3.3 Why Static Prompt Rules Backfire (Live Specimen of Skill Contamination)

1. **Accuracy prompt (`e4b_think_hint`) shifts rather than removes bugs**: `pass@1` barely moves (`22.4% -> 24.2%`), while `geopy`/geodesic attempts increase swapped `(lon, lat)` errors (`28 -> 41`), degree-space KD-tree nearest-neighbor bugs (`12 -> 30`), and general programming bugs (`27 -> 66`).
2. **Explicit 4-rule prompt (`e4b_think_rules`) causes severe contamination on unrelated tasks**:
   - Projection errors drop `50.4% -> 2.0%` and `lucky` drops `26.9% -> 0.3%`, **but runtime crashes double (`21.4% -> 41.8%`)**.
   - Median script length nearly doubles (`44 -> 80` lines) as the model forces `estimate_utm_crs` / `Geod` into 3,833/3,840 answers — causing non-projection tasks to collapse: `points_in_polygon` (`78.7% -> 41.9%`), `area_weighted_population` (`43.8% -> 8.7%`), `mean_per_district` (`22.5% -> 13.8%`).
   - **Swallowed exceptions (`scripts/caught_exceptions.py` via `sys.monitoring`)**: among runnable-but-wrong scripts under explicit rules, **26% (143/546)** caught a runtime exception (`AttributeError`, `GeodError`, `CRSError`) and printed `0` anyway — vs only **0.8% (18/2,159)** without rules.
3. **Contrast with `meta-rsi`**: Our metamorphic-gated evolved 4B no-think harness avoids static rule bloat (v0.2 4-seed `78.4 ± 3.8%` Test ALL, `8.9%` error, `2.1%` lucky; zero-leakage v4 `meta_neutral` up to `84.6%`, and co-evolved polar harness `87.3%` Test ALL, `1.7%` error, `0.6%` lucky), showing why execution-verified structural harness evolution outperforms hand-written prompt rules.

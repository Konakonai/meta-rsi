# gis-silent-bench

Synthetic GIS analysis tasks for studying silent errors: code that runs but returns a wrong number.
Each task is written three times with the same local geometry at three places on the globe, and a script must be right on all three to count as correct.
This catches methods that are right only by location, for example measuring in EPSG:3857 near the equator or hard-coding one UTM zone.

## Layout

- `gsb/geo.py`: location variants and placement of local-metre geometry on WGS84.
- `gsb/families.py`: task families (geometry generator, question, reference solution).
- `gsb/build.py`: writes data files, prompts and ground truth (`tasks.jsonl`).
- `gsb/harness.py`: extracts code from a model answer, runs it on the three variants, classifies the result.
- `gsb/audit.py`: re-solves every task with known right and wrong methods to show what the scoring can catch.
- `scripts/zero_shot.py`: samples answers with vLLM and scores them.
- `scripts/classify_errors.py`: labels silent_wrong / lucky answers by matching them against re-solved wrong methods (EPSG:3857, degrees, fixed UTM zone, one zone over a large extent, unit factor).
- `scripts/classify_llm.py`: labels the same answers with the Gemini Decisions API (`GEMINI_API_KEY`), for agreement with the rules and for answers the rules cannot place.

## Variants

| Variant | Center | Why |
|---|---|---|
| `low` | latitude within ±7° | EPSG:3857 distortion under 1%, so 3857 passes here by luck |
| `high` | latitude ±52–64° | EPSG:3857 inflates lengths 1.6–2.3× |
| `zone_edge` | latitude ±25–45°, on a UTM zone boundary | data straddles two zones; far from any fixed zone |

Large-extent families use `low` (±5°), `mid` (±28–40°) and `zone_edge` (±28–40°, on a zone boundary), shift degree-based geometry instead of projecting it, and take ground truth from `pyproj.Geod` (ellipsoidal area and distance).

Ground truth for local families is computed in an azimuthal equidistant CRS centered on the data.
The prompt never states the location or CRS. GeoPackage layers are stored in a local UTM CRS in 25% of tasks, and point layers can be CSV with `lon`/`lat` columns.

## Families

Seen (train and test):

- Projection-sensitive, local extent: `buffer_union_area`, `count_within`, `mean_nn_distance`, `sum_within`, `nearest_facility_max`, `area_by_class`, `perimeter_by_class`, `overlay_area`, `centroid_distance`, `road_length_by_type`, `count_near_roads`, `roads_in_district`.
- Method traps that do not depend on the projection: `points_in_polygon`, `area_weighted_population` (overlay without scaling by the clipped share), `covered_population` (overlapping buffers count a community twice), `mean_per_district` (an inner join drops empty districts). `roads_in_district` also separates `within`, `intersects` and clipping.
- Large extent (tens of degrees; one UTM zone is no longer accurate, geodesic or equal-area methods needed): `large_region_area`, `large_route_length`.

Held out (test only): `density_in_polygon`, `road_buffer_area`, `max_pair_distance`.

Answers are in metres, kilometres, square metres, square kilometres or hectares, chosen per task.

## Scoring

Real-valued answers must be within 1% of the truth, counts must match exactly.

| Category | Meaning |
|---|---|
| `correct` | right on all three variants |
| `error` | exception, timeout, or last output line is not a number on some variant |
| `silent_wrong` | runs everywhere, wrong everywhere |
| `lucky` | runs everywhere, right on some variants only |
| `no_code` | no Python code block in the answer |

## Usage

    nix-shell
    python -m gsb.build --out data/v1 --train 150 --test 20 --heldout 40 --seed 0
    python -m gsb.audit data/v1
    python -m pytest -q tests

## Known limits

- Synthetic data and English prompts; each answer is a single number.
- `points_in_polygon` does not depend on the projection, so it tests filtering and spatial predicates rather than CRS handling.
- Count tasks only catch a distorted projection when some features lie between the distorted and the true threshold distance.
- Buffer-area tasks only separate methods when buffers overlap (without overlap the 3857 answer equals the true one), so `buffer_union_area` keeps union/sum area between 0.3 and 0.8.

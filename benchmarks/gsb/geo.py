"""Placing locally built geometry on the globe, and the three location variants."""
from __future__ import annotations

import numpy as np
import shapely
from pyproj import CRS, Transformer

# Each task is written three times with the same local geometry at different places.
# low: Web Mercator distortion < 1%, so a 3857 solution can pass here by luck.
# high: 3857 inflates distances by 1/cos(lat) ~ 1.6-2.3x.
# zone_edge: data straddles a UTM zone boundary at mid latitude, far from any fixed zone.
VARIANTS = ("low", "high", "zone_edge")
# Large-extent tasks span tens of degrees, so they skip high latitudes and stay off the antimeridian.
LARGE_VARIANTS = ("low", "mid", "zone_edge")


def sample_center(variant: str, rng: np.random.Generator) -> tuple[float, float]:
    """Return (lat, lon) of the data center for one variant."""
    if variant == "low":
        return float(rng.uniform(-7, 7)), float(rng.uniform(-178, 178))
    if variant == "high":
        return float(rng.choice([-1, 1]) * rng.uniform(52, 64)), float(rng.uniform(-178, 178))
    if variant == "zone_edge":
        lat = float(rng.choice([-1, 1]) * rng.uniform(25, 45))
        lon = -180.0 + 6.0 * int(rng.integers(1, 60))  # a UTM zone boundary
        return lat, lon
    raise ValueError(variant)


def sample_center_large(variant: str, rng: np.random.Generator) -> tuple[float, float]:
    if variant == "low":
        return float(rng.uniform(-5, 5)), float(rng.uniform(-150, 150))
    if variant == "mid":
        return float(rng.choice([-1, 1]) * rng.uniform(28, 40)), float(rng.uniform(-150, 150))
    if variant == "zone_edge":
        return float(rng.choice([-1, 1]) * rng.uniform(28, 40)), -180.0 + 6.0 * int(rng.integers(6, 55))
    raise ValueError(variant)


def local_crs(lat: float, lon: float) -> CRS:
    """Azimuthal equidistant CRS centered on (lat, lon); metres, accurate for small extents."""
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def place(geoms, lat: float, lon: float):
    """Map shapely geometries in local metres (origin = center) to WGS84 lon/lat."""
    tr = Transformer.from_crs(local_crs(lat, lon), "EPSG:4326", always_xy=True)

    def f(coords):
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return [shapely.transform(g, f) for g in geoms]

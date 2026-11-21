"""Global placement, circular antimeridian-safe AEQD centering, and physical GeoTIFF synthesis for GSB v2."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

# Strict weak proxy (|lat| <= 2.2 deg => EPSG:3857 length distortion <= 0.074%, area <= 0.148%)
VARIANTS_LOCAL = ("low", "high", "polar", "antimeridian")
VARIANTS_RASTER = ("low", "high", "polar")
VARIANTS_LARGE = ("low", "mid", "zone_edge")


def sample_center(variant: str, rng: np.random.Generator) -> tuple[float, float]:
    """Return (lat, lon) center for local vector or raster variants."""
    if variant == "low":
        return float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-170.0, 170.0))
    if variant == "high":
        return float(rng.choice([-1.0, 1.0]) * rng.uniform(54.0, 64.0)), float(rng.uniform(-170.0, 170.0))
    if variant == "polar":
        if rng.random() < 0.5:
            return float(rng.uniform(85.3, 87.2)), float(rng.uniform(-170.0, 170.0))
        return float(-rng.uniform(85.3, 87.2)), float(rng.uniform(-170.0, 170.0))
    if variant == "antimeridian":
        return float(rng.choice([-1.0, 1.0]) * rng.uniform(15.0, 42.0)), 180.0
    raise ValueError(f"Unknown local variant: {variant}")


def sample_center_large(variant: str, rng: np.random.Generator) -> tuple[float, float]:
    if variant == "low":
        return float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-140.0, 140.0))
    if variant == "mid":
        return float(rng.choice([-1.0, 1.0]) * rng.uniform(28.0, 42.0)), float(rng.uniform(-140.0, 140.0))
    if variant == "zone_edge":
        return float(rng.choice([-1.0, 1.0]) * rng.uniform(28.0, 42.0)), -180.0 + 6.0 * int(rng.integers(6, 55))
    raise ValueError(f"Unknown large variant: {variant}")


def local_crs(lat: float, lon: float) -> CRS:
    """Azimuthal equidistant CRS centered on (lat, lon) in WGS84 metres."""
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def place(geoms: list, lat: float, lon: float) -> list:
    """Map local-metre shapely geometries around origin to WGS84 lon/lat via local AEQD."""
    tr = Transformer.from_crs(local_crs(lat, lon), "EPSG:4326", always_xy=True)

    def _f(coords):
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return [shapely.transform(g, _f) for g in geoms]


def circular_center_wgs84(layers: dict[str, gpd.GeoDataFrame], rasters: dict[str, Path] | None = None) -> tuple[float, float]:
    """Compute spherical unit-vector centroid (lat, lon) across all vector & raster layers.

    Unlike naive `(minx + maxx) / 2`, this stays exact across the antimeridian (+/-180 deg)
    and near the poles.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    def _add_pts(lons: np.ndarray, lats: np.ndarray) -> None:
        if len(lons) == 0:
            return
        rlon = np.deg2rad(lons)
        rlat = np.deg2rad(lats)
        clat = np.cos(rlat)
        xs.append(float(np.mean(clat * np.cos(rlon))))
        ys.append(float(np.mean(clat * np.sin(rlon))))
        zs.append(float(np.mean(np.sin(rlat))))

    for g in layers.values():
        if g.empty:
            continue
        g4326 = g.to_crs("EPSG:4326") if g.crs is not None else g.set_crs("EPSG:4326")
        pts = g4326.geometry.representative_point()
        _add_pts(pts.x.to_numpy(dtype=float), pts.y.to_numpy(dtype=float))

    if rasters:
        for rpath in rasters.values():
            with rasterio.open(rpath) as src:
                b = src.bounds
                cx = 0.5 * (b.left + b.right)
                cy = 0.5 * (b.bottom + b.top)
                tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                lon, lat = tr.transform(cx, cy)
                _add_pts(np.array([lon]), np.array([lat]))

    if not xs:
        return 0.0, 0.0
    mx, my, mz = float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))
    hyp = math.hypot(mx, my)
    lat_c = math.degrees(math.atan2(mz, hyp))
    lon_c = math.degrees(math.atan2(my, mx))
    return lat_c, lon_c


def reference_aeqd_crs(layers: dict[str, gpd.GeoDataFrame], rasters: dict[str, Path] | None = None) -> CRS:
    lat_c, lon_c = circular_center_wgs84(layers, rasters)
    return local_crs(lat_c, lon_c)


def storage_crs_for_raster(variant: str, lat: float, *, force_4326_high: bool = False) -> tuple[CRS, float]:
    """Choose GeoTIFF storage CRS and nodata sentinel for a given variant."""
    if variant == "low":
        return CRS.from_epsg(3857), -9999.0
    if variant == "high":
        if force_4326_high:
            return CRS.from_epsg(4326), -32768.0
        return CRS.from_epsg(3857), -32768.0
    if variant == "polar":
        # EPSG:3857 is undefined above 85.05N; store in EPSG:4326 or Polar Stereographic
        return (CRS.from_epsg(3413) if lat >= 0 else CRS.from_epsg(3031)), -9999.0
    return CRS.from_epsg(4326), -9999.0


def write_physical_geotiff(
    out_path: Path,
    *,
    lat: float,
    lon: float,
    extent_m: float,
    pixel_m: float,
    field_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    valid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    dst_crs: CRS,
    nodata: float,
) -> None:
    """Synthesize a GeoTIFF in `dst_crs` by evaluating an exact local-metre physical field `field_fn(x_m, y_m)`.

    Because every pixel center in `dst_crs` is transformed directly to local AEQD metres before
    evaluating `field_fn`, there is zero inter-variant resampling blur while preserving exact
    physical geometry (including EPSG:3857 scale distortion at high latitudes).
    """
    aeqd = local_crs(lat, lon)
    to_dst = Transformer.from_crs(aeqd, dst_crs, always_xy=True)
    to_aeqd = Transformer.from_crs(dst_crs, aeqd, always_xy=True)

    # Sample perimeter in local AEQD to get tight bounding box in dst_crs
    edge = np.linspace(-extent_m, extent_m, 25)
    bx = np.concatenate([edge, edge, np.full_like(edge, -extent_m), np.full_like(edge, extent_m)])
    by = np.concatenate([np.full_like(edge, -extent_m), np.full_like(edge, extent_m), edge, edge])
    dx, dy = to_dst.transform(bx, by)
    left, right = float(np.min(dx)), float(np.max(dx))
    bottom, top = float(np.min(dy)), float(np.max(dy))

    # Choose grid resolution proportional to true physical extent (~2*extent_m / pixel_m cells per side)
    n_pix = max(32, int(round(2.0 * extent_m / pixel_m)))
    width = n_pix
    height = n_pix
    transform = from_bounds(left, bottom, right, top, width, height)

    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    gx = transform.c + cc * transform.a + rr * transform.b
    gy = transform.f + cc * transform.d + rr * transform.e

    xm, ym = to_aeqd.transform(gx, gy)
    vals = field_fn(xm, ym).astype(np.float32)

    # Mark pixels outside [-extent_m, extent_m]^2 or masked by valid_fn as nodata
    inside = (np.abs(xm) <= extent_m * 1.001) & (np.abs(ym) <= extent_m * 1.001)
    if valid_fn is not None:
        inside = inside & valid_fn(xm, ym)
    vals[~inside] = np.float32(nodata)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=dst_crs,
        transform=transform,
        nodata=float(nodata),
        compress="deflate",
    ) as dst:
        dst.write(vals, 1)


def reproject_raster_to_crs(
    src_path: Path,
    dst_crs: CRS,
    *,
    resampling: Resampling = Resampling.bilinear,
    target_res_m: float | None = None,
) -> tuple[np.ndarray, rasterio.Affine, float]:
    """Reproject a single-band GeoTIFF to `dst_crs`, preserving nodata cleanly."""
    with rasterio.open(src_path) as src:
        nodata = float(src.nodata) if src.nodata is not None else -9999.0
        if target_res_m is not None:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_res_m
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
        dst_arr = np.full((height, width), np.float32(nodata), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=nodata,
            resampling=resampling,
        )
    return dst_arr, transform, nodata

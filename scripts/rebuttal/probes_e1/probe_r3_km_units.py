from pathlib import Path
import shutil

import geopandas as gpd
import pandas as pd
from pyproj import CRS
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject


def _estimate_center(src_dir: Path) -> tuple[float, float]:
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".gpkg", ".geojson"}:
                gdf = gpd.read_file(path)
                if not gdf.empty:
                    gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf
                    minx, miny, maxx, maxy = gdf_wgs.total_bounds
                    return float((minx + maxx) / 2.0), float((miny + maxy) / 2.0)
            elif suffix == ".tif":
                with rasterio.open(path) as src:
                    if src.crs is not None:
                        gdf = gpd.GeoDataFrame(
                            geometry=gpd.points_from_xy(
                                [(src.bounds.left + src.bounds.right) / 2.0],
                                [(src.bounds.bottom + src.bounds.top) / 2.0],
                            ),
                            crs=src.crs,
                        ).to_crs("EPSG:4326")
                        pt = gdf.geometry.iloc[0]
                        return float(pt.x), float(pt.y)
            elif suffix == ".csv":
                df = pd.read_csv(path)
                if not df.empty and "lon" in df.columns and "lat" in df.columns:
                    return float(df["lon"].mean()), float(df["lat"].mean())
        except Exception:
            continue
    return 0.0, 0.0


def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    lon_0, lat_0 = _estimate_center(src_dir)
    target_crs = CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat_0:.6f} +lon_0={lon_0:.6f} "
        "+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=km +no_defs"
    )

    transformed_files = []
    for src_path in sorted(src_dir.rglob("*")):
        if not src_path.is_file():
            continue
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = src_path.suffix.lower()
        if suffix == ".gpkg":
            gdf = gpd.read_file(src_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            gdf_proj = gdf.to_crs(target_crs)
            gdf_proj.to_file(dst_path, driver="GPKG")
            transformed_files.append(str(rel_path))
        elif suffix == ".tif":
            with rasterio.open(src_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update(
                    {
                        "crs": target_crs.to_wkt(),
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )
                with rasterio.open(dst_path, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest,
                        )
            transformed_files.append(str(rel_path))
        else:
            shutil.copy2(src_path, dst_path)

    return {
        "mode": "projected_km_units",
        "target_crs": target_crs.to_proj4(),
        "transformed_files": transformed_files,
        "seed": seed,
    }
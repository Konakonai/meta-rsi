from pathlib import Path
import shutil
import geopandas as gpd


def perturb(src_dir: Path, dst_dir: Path, seed: int = 42) -> dict:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    transformed_gpkg = 0
    copied_other = 0

    for src_path in sorted(src_dir.rglob("*")):
        if src_path.is_dir():
            continue
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix.lower() == ".gpkg":
            gdf = gpd.read_file(src_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            # Check latitude bounds in WGS84 before projecting to Web Mercator
            gdf_4326 = gdf.to_crs("EPSG:4326")
            if not gdf_4326.empty:
                _, miny, _, maxy = gdf_4326.total_bounds
                if -85.0 <= miny and maxy <= 85.0:
                    gdf_out = gdf.to_crs("EPSG:3857")
                    gdf_out.to_file(dst_path, driver="GPKG")
                    transformed_gpkg += 1
                    continue

            shutil.copy2(src_path, dst_path)
            copied_other += 1
        else:
            shutil.copy2(src_path, dst_path)
            copied_other += 1

    return {
        "mode": "gpkg_web_mercator_crs",
        "transformed_gpkg": transformed_gpkg,
        "copied_other": copied_other,
    }
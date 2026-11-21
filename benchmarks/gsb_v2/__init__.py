"""GSB v2: Multi-modal (vector + raster/DEM), 4-regime (low/high/polar/antimeridian) Geospatial Silent-Error Benchmark."""
from .families import BY_NAME, FAMILIES, Family, Spec
from .geo import VARIANTS_LARGE, VARIANTS_LOCAL, VARIANTS_RASTER

__all__ = ["BY_NAME", "FAMILIES", "Family", "Spec", "VARIANTS_LARGE", "VARIANTS_LOCAL", "VARIANTS_RASTER"]

"""Exact invertible physical unit, calibration, and missing-sentinel transforms for SciUnits."""
from __future__ import annotations

import numpy as np

# Each entry maps unit_name -> (dimension, to_si_fn, from_si_fn)
# All affine/linear conversions are exact and invertible to machine precision.
UNIT_TABLE: dict[str, tuple[str, tuple[float, float]]] = {
    # (dimension, (scale_to_si, offset_to_si)) where si = val * scale + offset
    # Length (SI: m)
    "m": ("length", (1.0, 0.0)),
    "km": ("length", (1000.0, 0.0)),
    "cm": ("length", (0.01, 0.0)),
    "mm": ("length", (0.001, 0.0)),
    "ft": ("length", (0.3048, 0.0)),
    "in": ("length", (0.0254, 0.0)),
    # Area (SI: m2)
    "m2": ("area", (1.0, 0.0)),
    "cm2": ("area", (1e-4, 0.0)),
    "ft2": ("area", (0.09290304, 0.0)),
    # Volume (SI: m3)
    "m3": ("volume", (1.0, 0.0)),
    "L": ("volume", (1e-3, 0.0)),
    "ft3": ("volume", (0.028316846592, 0.0)),
    "gal_us": ("volume", (0.003785411784, 0.0)),
    # Mass (SI: kg)
    "kg": ("mass", (1.0, 0.0)),
    "g": ("mass", (1e-3, 0.0)),
    "t": ("mass", (1000.0, 0.0)),
    "lb": ("mass", (0.45359237, 0.0)),
    # Time (SI: s)
    "s": ("time", (1.0, 0.0)),
    "min": ("time", (60.0, 0.0)),
    "hr": ("time", (3600.0, 0.0)),
    # Speed (SI: m/s)
    "m/s": ("speed", (1.0, 0.0)),
    "km/h": ("speed", (1.0 / 3.6, 0.0)),
    "mph": ("speed", (0.44704, 0.0)),
    "ft/s": ("speed", (0.3048, 0.0)),
    # Temperature (SI: K)
    "K": ("temperature", (1.0, 0.0)),
    "degC": ("temperature", (1.0, 273.15)),
    "degF": ("temperature", (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0)),
    # Pressure (SI: Pa)
    "Pa": ("pressure", (1.0, 0.0)),
    "kPa": ("pressure", (1000.0, 0.0)),
    "bar": ("pressure", (1e5, 0.0)),
    "psi": ("pressure", (6894.757293168, 0.0)),
    # Force (SI: N)
    "N": ("force", (1.0, 0.0)),
    "kN": ("force", (1000.0, 0.0)),
    "lbf": ("force", (4.4482216152605, 0.0)),
    # Energy (SI: J)
    "J": ("energy", (1.0, 0.0)),
    "kJ": ("energy", (1000.0, 0.0)),
    "MJ": ("energy", (1e6, 0.0)),
    "kWh": ("energy", (3.6e6, 0.0)),
    "kcal": ("energy", (4184.0, 0.0)),
    # Power (SI: W)
    "W": ("power", (1.0, 0.0)),
    "kW": ("power", (1000.0, 0.0)),
    "MW": ("power", (1e6, 0.0)),
    "hp": ("power", (745.6998715822702, 0.0)),
    # Irradiance (SI: W/m2)
    "W/m2": ("irradiance", (1.0, 0.0)),
    "kW/m2": ("irradiance", (1000.0, 0.0)),
    # Density (SI: kg/m3)
    "kg/m3": ("density", (1.0, 0.0)),
    "g/cm3": ("density", (1000.0, 0.0)),
    "lb/ft3": ("density", (16.01846337396014, 0.0)),
    # Specific heat capacity (SI: J/(kg*K))
    "J/(kg*K)": ("spec_heat", (1.0, 0.0)),
    "kJ/(kg*K)": ("spec_heat", (1000.0, 0.0)),
    # Second moment of area (SI: m4)
    "m4": ("inertia4", (1.0, 0.0)),
    "cm4": ("inertia4", (1e-8, 0.0)),
    "in4": ("inertia4", (0.0254 ** 4, 0.0)),
    # Angle (SI: rad)
    "rad": ("angle", (1.0, 0.0)),
    "deg": ("angle", (float(np.pi / 180.0), 0.0)),
    # Concentration (SI: kg/m3 == g/L)
    "g/L": ("concentration", (1.0, 0.0)),
    "mg/L": ("concentration", (1e-3, 0.0)),
    "kg/L": ("concentration", (1000.0, 0.0)),
    # Dimensionless ratio / fraction
    "fraction": ("ratio", (1.0, 0.0)),
    "percent": ("ratio", (0.01, 0.0)),
}

ALT_UNITS_BY_DIM: dict[str, list[str]] = {}
SI_UNIT_BY_DIM: dict[str, str] = {}
for _u, (_dim, (_sc, _off)) in UNIT_TABLE.items():
    ALT_UNITS_BY_DIM.setdefault(_dim, []).append(_u)
    if abs(_sc - 1.0) < 1e-12 and abs(_off) < 1e-12 and _dim not in SI_UNIT_BY_DIM:
        SI_UNIT_BY_DIM[_dim] = _u


def to_si(values: np.ndarray | float, unit: str) -> np.ndarray | float:
    _, (scale, offset) = UNIT_TABLE[unit]
    return values * scale + offset


def from_si(si_values: np.ndarray | float, unit: str) -> np.ndarray | float:
    _, (scale, offset) = UNIT_TABLE[unit]
    return (si_values - offset) / scale


def convert_unit(values: np.ndarray | float, src_unit: str, dst_unit: str) -> np.ndarray | float:
    d1, _ = UNIT_TABLE[src_unit]
    d2, _ = UNIT_TABLE[dst_unit]
    if d1 != d2:
        raise ValueError(f"Incompatible dimensions: {src_unit} ({d1}) vs {dst_unit} ({d2})")
    return from_si(to_si(values, src_unit), dst_unit)


def pick_alt_unit(si_unit: str, rng: np.random.Generator, variant_idx: int = 0) -> str:
    dim, _ = UNIT_TABLE[si_unit]
    alts = [u for u in ALT_UNITS_BY_DIM[dim] if u != si_unit]
    if not alts:
        return si_unit
    return str(alts[(int(rng.integers(0, len(alts))) + variant_idx) % len(alts)])

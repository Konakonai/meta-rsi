"""12 Scientific & Engineering Task Families for SciUnits (8 seen + 4 held-out).

Every family produces realistic multi-table CSV + `metadata.json` workspaces where:
- `low` stores all quantities in canonical SI units, calibration (scale=1, offset=0),
  empty/NaN missing values (`missing_sentinel=null`), and sorted timestamps/IDs.
- Oracle variants (`unit_imperial`, `sentinel_shifted`, `permuted_mixed`) apply exact
  invertible physical unit conversions, ADC/sensor linear calibration (`raw * scale + offset`),
  non-null missing-value sentinels (`-999.0` / `-9999.0`), row/column permutations, and
  unrelated NaN auxiliary columns while keeping the physical answer strictly invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

VARIANTS_SCI = ("low", "unit_imperial", "sentinel_shifted", "permuted_mixed")

G_ACCEL = 9.80665
R_GAS = 8.314462618


@dataclass
class TableSpec:
    df: pd.DataFrame
    si_units: dict[str, str] = field(default_factory=dict)
    calibratable: list[str] = field(default_factory=list)
    nullable_cols: list[str] = field(default_factory=list)
    time_col: str | None = None
    description: str = ""


@dataclass
class SciSpec:
    params: dict
    tables: dict[str, TableSpec]


@dataclass
class SciFamily:
    name: str
    group: str  # "thermo", "mech", "timeseries", "control"
    held_out: bool
    answer: str  # "float" or "count"
    build: Callable[[np.random.Generator], SciSpec]
    solve: Callable[[dict[str, pd.DataFrame], dict], float]
    question: Callable[[dict], str]


def _inject_nans(rng: np.random.Generator, arr: np.ndarray, frac: float = 0.12) -> np.ndarray:
    out = arr.astype(float).copy()
    n_miss = max(1, int(round(len(out) * frac)))
    if len(out) > n_miss + 2:
        idx = rng.choice(len(out), size=n_miss, replace=False)
        out[idx] = np.nan
    return out


# =============================================================================
# 1. thermal_energy_budget (seen, thermo)
# =============================================================================
def _build_thermal_energy_budget(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(14, 22))
    zone = list(rng.choice(["core", "jacket", "aux"], n))
    for i in range(4):
        zone[i] = "core"
    target_zone = str(rng.choice(["core", "jacket"]))
    for i in range(4):
        zone[i] = target_zone
    mass = rng.uniform(50.0, 400.0, n)
    spec_heat = rng.uniform(450.0, 4200.0, n)
    t_init = rng.uniform(285.0, 315.0, n)
    t_final = t_init + rng.uniform(15.0, 85.0, n)
    t_final = _inject_nans(rng, t_final, 0.14)
    # Ensure >= 3 valid target_zone rows
    for i in range(3):
        if np.isnan(t_final[i]):
            t_final[i] = t_init[i] + 35.0
    df = pd.DataFrame({
        "sample_id": np.arange(n),
        "zone": zone,
        "mass": np.round(mass, 4),
        "spec_heat": np.round(spec_heat, 4),
        "temp_init": np.round(t_init, 4),
        "temp_final": np.round(t_final, 4),
    })
    return SciSpec(
        params={"zone": target_zone},
        tables={
            "calorimetry": TableSpec(
                df=df,
                si_units={"mass": "kg", "spec_heat": "J/(kg*K)", "temp_init": "K", "temp_final": "K"},
                calibratable=["mass"],
                nullable_cols=["temp_final"],
                description="calorimetry batches (`sample_id`, `zone`, `mass`, `spec_heat`, `temp_init`, `temp_final`)",
            )
        },
    )


def _solve_thermal_energy_budget(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["calorimetry"]
    sub = df[df["zone"] == params["zone"]].dropna(subset=["mass", "spec_heat", "temp_init", "temp_final"])
    q_j = float(np.sum(sub["mass"] * sub["spec_heat"] * (sub["temp_final"] - sub["temp_init"])))
    return q_j / 1e6  # MJ


# =============================================================================
# 2. hydrostatic_dam_force (seen, mech)
# =============================================================================
def _build_hydrostatic_dam_force(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(12, 20))
    status = list(rng.choice(["open", "closed"], n))
    for i in range(4):
        status[i] = "closed"
    p_atm = np.full(n, 101325.0) + rng.uniform(-1500.0, 1500.0, n)
    density = rng.uniform(995.0, 1035.0, n)
    depth = _inject_nans(rng, rng.uniform(4.0, 45.0, n), 0.12)
    area = rng.uniform(1.5, 18.0, n)
    for i in range(3):
        if np.isnan(depth[i]):
            depth[i] = 15.0
    df = pd.DataFrame({
        "gate_id": np.arange(n),
        "status": status,
        "p_atm": np.round(p_atm, 2),
        "fluid_density": np.round(density, 4),
        "depth": np.round(depth, 4),
        "gate_area": np.round(area, 4),
    })
    return SciSpec(
        params={"status": "closed"},
        tables={
            "gates": TableSpec(
                df=df,
                si_units={"p_atm": "Pa", "fluid_density": "kg/m3", "depth": "m", "gate_area": "m2"},
                calibratable=["depth"],
                nullable_cols=["depth"],
                description="submerged dam gates (`gate_id`, `status`, `p_atm`, `fluid_density`, `depth`, `gate_area`)",
            )
        },
    )


def _solve_hydrostatic_dam_force(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["gates"]
    sub = df[df["status"] == params["status"]].dropna(subset=["p_atm", "fluid_density", "depth", "gate_area"])
    pressure = sub["p_atm"] + sub["fluid_density"] * G_ACCEL * sub["depth"]
    return float(np.sum(pressure * sub["gate_area"])) / 1e3  # kN


# =============================================================================
# 3. beam_bending_stress (seen, mech)
# =============================================================================
def _build_beam_bending_stress(rng: np.random.Generator) -> SciSpec:
    n_sec = int(rng.integers(5, 9))
    sec_df = pd.DataFrame({
        "section_code": [f"S{i}" for i in range(n_sec)],
        "half_depth": np.round(rng.uniform(0.08, 0.35, n_sec), 5),
        "inertia": np.round(rng.uniform(2e-5, 5e-4, n_sec), 8),
    })
    n_beams = int(rng.integers(14, 22))
    loads = _inject_nans(rng, rng.uniform(5000.0, 45000.0, n_beams), 0.12)
    spans = rng.uniform(2.5, 12.0, n_beams)
    for i in range(3):
        if np.isnan(loads[i]):
            loads[i] = 18000.0
    beam_df = pd.DataFrame({
        "beam_id": np.arange(n_beams),
        "section_code": rng.choice(sec_df["section_code"], n_beams),
        "point_load": np.round(loads, 3),
        "span_length": np.round(spans, 4),
    })
    return SciSpec(
        params={},
        tables={
            "beams": TableSpec(
                df=beam_df,
                si_units={"point_load": "N", "span_length": "m"},
                calibratable=["point_load"],
                nullable_cols=["point_load"],
                description="simply-supported beams (`beam_id`, `section_code`, `point_load`, `span_length`)",
            ),
            "sections": TableSpec(
                df=sec_df,
                si_units={"half_depth": "m", "inertia": "m4"},
                calibratable=[],
                nullable_cols=[],
                description="cross-section properties (`section_code`, `half_depth`, `inertia`)",
            ),
        },
    )


def _solve_beam_bending_stress(tables: dict[str, pd.DataFrame], params: dict) -> float:
    m = tables["beams"].merge(tables["sections"], on="section_code", how="inner")
    m = m.dropna(subset=["point_load", "span_length", "half_depth", "inertia"])
    # Simply supported beam with central point load P: M_max = P * L / 4; sigma = M_max * y / I
    moment = m["point_load"] * m["span_length"] / 4.0
    sigma_pa = moment * m["half_depth"] / m["inertia"]
    return float(sigma_pa.max()) / 1e6  # MPa


# =============================================================================
# 4. trapezoidal_power_energy (seen, timeseries)
# =============================================================================
def _build_trapezoidal_power_energy(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(24, 40))
    dt = rng.uniform(120.0, 900.0, n - 1)
    t_s = np.concatenate([[0.0], np.cumsum(dt)])
    power_w = rng.uniform(800.0, 12000.0, n)
    power_w = _inject_nans(rng, power_w, 0.12)
    power_w[0] = 2500.0
    power_w[-1] = 3200.0
    df = pd.DataFrame({
        "reading_id": np.arange(n),
        "elapsed_time": np.round(t_s, 4),
        "power": np.round(power_w, 4),
    })
    return SciSpec(
        params={},
        tables={
            "power_log": TableSpec(
                df=df,
                si_units={"elapsed_time": "s", "power": "W"},
                calibratable=["power"],
                nullable_cols=["power"],
                time_col="elapsed_time",
                description="power meter time-series (`reading_id`, `elapsed_time`, `power`)",
            )
        },
    )


def _solve_trapezoidal_power_energy(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["power_log"].dropna(subset=["elapsed_time", "power"]).sort_values("elapsed_time")
    e_j = float(np.trapezoid(df["power"].to_numpy(dtype=float), df["elapsed_time"].to_numpy(dtype=float)))
    return e_j / 3.6e6  # kWh


# =============================================================================
# 5. ideal_gas_moles_filtered (seen, thermo)
# =============================================================================
def _build_ideal_gas_moles_filtered(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(14, 24))
    gas = list(rng.choice(["N2", "Ar", "He"], n))
    target_gas = str(rng.choice(["N2", "Ar"]))
    for i in range(4):
        gas[i] = target_gas
    pressure = rng.uniform(1.2e5, 8.5e5, n)
    volume = rng.uniform(0.05, 1.8, n)
    temp_k = _inject_nans(rng, rng.uniform(270.0, 420.0, n), 0.14)
    for i in range(3):
        if np.isnan(temp_k[i]):
            temp_k[i] = 310.0
    df = pd.DataFrame({
        "vessel_id": np.arange(n),
        "gas_type": gas,
        "pressure": np.round(pressure, 3),
        "volume": np.round(volume, 5),
        "temperature": np.round(temp_k, 4),
    })
    return SciSpec(
        params={"gas_type": target_gas},
        tables={
            "vessels": TableSpec(
                df=df,
                si_units={"pressure": "Pa", "volume": "m3", "temperature": "K"},
                calibratable=["pressure"],
                nullable_cols=["temperature"],
                description="pressurized gas vessels (`vessel_id`, `gas_type`, `pressure`, `volume`, `temperature`)",
            )
        },
    )


def _solve_ideal_gas_moles_filtered(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["vessels"]
    sub = df[df["gas_type"] == params["gas_type"]].dropna(subset=["pressure", "volume", "temperature"])
    moles = (sub["pressure"] * sub["volume"]) / (R_GAS * sub["temperature"])
    return float(moles.sum())


# =============================================================================
# 6. kinetic_braking_work (seen, mech)
# =============================================================================
def _build_kinetic_braking_work(rng: np.random.Generator) -> SciSpec:
    n_models = int(rng.integers(4, 8))
    cat_df = pd.DataFrame({
        "model_code": [f"V{i}" for i in range(n_models)],
        "vehicle_mass": np.round(rng.uniform(1200.0, 18000.0, n_models), 2),
    })
    n_events = int(rng.integers(14, 24))
    v1 = rng.uniform(18.0, 38.0, n_events)
    v2 = _inject_nans(rng, v1 * rng.uniform(0.1, 0.55, n_events), 0.12)
    for i in range(3):
        if np.isnan(v2[i]):
            v2[i] = v1[i] * 0.3
    ev_df = pd.DataFrame({
        "event_id": np.arange(n_events),
        "model_code": rng.choice(cat_df["model_code"], n_events),
        "v_initial": np.round(v1, 4),
        "v_final": np.round(v2, 4),
    })
    return SciSpec(
        params={},
        tables={
            "braking_events": TableSpec(
                df=ev_df,
                si_units={"v_initial": "m/s", "v_final": "m/s"},
                calibratable=["v_initial"],
                nullable_cols=["v_final"],
                description="braking telemetry events (`event_id`, `model_code`, `v_initial`, `v_final`)",
            ),
            "vehicles": TableSpec(
                df=cat_df,
                si_units={"vehicle_mass": "kg"},
                calibratable=[],
                nullable_cols=[],
                description="vehicle mass specifications (`model_code`, `vehicle_mass`)",
            ),
        },
    )


def _solve_kinetic_braking_work(tables: dict[str, pd.DataFrame], params: dict) -> float:
    m = tables["braking_events"].merge(tables["vehicles"], on="model_code", how="inner")
    m = m.dropna(subset=["vehicle_mass", "v_initial", "v_final"])
    dk_j = 0.5 * m["vehicle_mass"] * (m["v_initial"] ** 2 - m["v_final"] ** 2)
    return float(dk_j.sum()) / 1e6  # MJ


# =============================================================================
# 7. pipe_darcy_headloss (seen, mech)
# =============================================================================
def _build_pipe_darcy_headloss(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(12, 20))
    line_id = list(rng.choice(["main", "bypass"], n))
    for i in range(4):
        line_id[i] = "main"
    fric = rng.uniform(0.015, 0.032, n)
    length = rng.uniform(80.0, 1200.0, n)
    diam = rng.uniform(0.08, 0.55, n)
    vel = _inject_nans(rng, rng.uniform(0.8, 4.2, n), 0.12)
    for i in range(3):
        if np.isnan(vel[i]):
            vel[i] = 2.0
    df = pd.DataFrame({
        "seg_id": np.arange(n),
        "line_group": line_id,
        "friction_factor": np.round(fric, 5),
        "pipe_length": np.round(length, 4),
        "pipe_diameter": np.round(diam, 5),
        "flow_velocity": np.round(vel, 4),
    })
    return SciSpec(
        params={"line_group": "main"},
        tables={
            "pipes": TableSpec(
                df=df,
                si_units={"pipe_length": "m", "pipe_diameter": "m", "flow_velocity": "m/s"},
                calibratable=["flow_velocity"],
                nullable_cols=["flow_velocity"],
                description="pipe network segments (`seg_id`, `line_group`, `friction_factor`, `pipe_length`, `pipe_diameter`, `flow_velocity`)",
            )
        },
    )


def _solve_pipe_darcy_headloss(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["pipes"]
    sub = df[df["line_group"] == params["line_group"]].dropna(
        subset=["friction_factor", "pipe_length", "pipe_diameter", "flow_velocity"]
    )
    hf = sub["friction_factor"] * (sub["pipe_length"] / sub["pipe_diameter"]) * (sub["flow_velocity"] ** 2) / (2.0 * G_ACCEL)
    return float(hf.sum())  # m


# =============================================================================
# 8. control_weighted_sensor_mean (seen, control — zero-entry & non-unit control)
# =============================================================================
def _build_control_weighted_sensor_mean(rng: np.random.Generator) -> SciSpec:
    n_sites = int(rng.integers(8, 13))
    sites = pd.DataFrame({
        "site_code": [f"Z{i}" for i in range(n_sites)],
        "region": rng.choice(["north", "south"], n_sites),
    })
    # Leave at least 2 sites with 0 readings to test zero-inclusion groupby mean
    active_codes = list(sites["site_code"].iloc[:-2])
    n_obs = int(rng.integers(20, 34))
    obs = pd.DataFrame({
        "obs_id": np.arange(n_obs),
        "site_code": rng.choice(active_codes, n_obs),
        "alert_count": rng.integers(1, 9, n_obs).astype(float),
    })
    return SciSpec(
        params={},
        tables={
            "sites": TableSpec(
                df=sites,
                si_units={},
                calibratable=[],
                nullable_cols=[],
                description="monitoring sites (`site_code`, `region`)",
            ),
            "alerts": TableSpec(
                df=obs,
                si_units={},
                calibratable=[],
                nullable_cols=[],
                description="alert counts per site (`obs_id`, `site_code`, `alert_count`)",
            ),
        },
    )


def _solve_control_weighted_sensor_mean(tables: dict[str, pd.DataFrame], params: dict) -> float:
    sites = tables["sites"]
    alerts = tables["alerts"]
    totals = alerts.groupby("site_code")["alert_count"].sum()
    per_site = sites["site_code"].map(totals).fillna(0.0)
    return float(per_site.mean())


# =============================================================================
# 9. solar_panel_yield_kwh (HELD-OUT, timeseries)
# =============================================================================
def _build_solar_panel_yield_kwh(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(20, 32))
    dt = rng.uniform(300.0, 1200.0, n - 1)
    t_s = np.concatenate([[0.0], np.cumsum(dt)])
    irr = _inject_nans(rng, rng.uniform(220.0, 950.0, n), 0.12)
    irr[0], irr[-1] = 400.0, 500.0
    area_m2 = round(float(rng.uniform(12.0, 65.0)), 3)
    eff = round(float(rng.uniform(0.16, 0.24)), 4)
    meta_df = pd.DataFrame({"array_id": [0], "panel_area": [area_m2], "efficiency": [eff]})
    log_df = pd.DataFrame({"step_id": np.arange(n), "elapsed_time": np.round(t_s, 4), "irradiance": np.round(irr, 4)})
    return SciSpec(
        params={},
        tables={
            "irradiance_log": TableSpec(
                df=log_df,
                si_units={"elapsed_time": "s", "irradiance": "W/m2"},
                calibratable=["irradiance"],
                nullable_cols=["irradiance"],
                time_col="elapsed_time",
                description="solar irradiance log (`step_id`, `elapsed_time`, `irradiance`)",
            ),
            "array_spec": TableSpec(
                df=meta_df,
                si_units={"panel_area": "m2", "efficiency": "fraction"},
                calibratable=[],
                nullable_cols=[],
                description="solar array specification (`array_id`, `panel_area`, `efficiency`)",
            ),
        },
    )


def _solve_solar_panel_yield_kwh(tables: dict[str, pd.DataFrame], params: dict) -> float:
    spec = tables["array_spec"].iloc[0]
    df = tables["irradiance_log"].dropna(subset=["elapsed_time", "irradiance"]).sort_values("elapsed_time")
    p_w = df["irradiance"].to_numpy(dtype=float) * float(spec["panel_area"]) * float(spec["efficiency"])
    e_j = float(np.trapezoid(p_w, df["elapsed_time"].to_numpy(dtype=float)))
    return e_j / 3.6e6  # kWh


# =============================================================================
# 10. carnot_heat_recovery (HELD-OUT, thermo)
# =============================================================================
def _build_carnot_heat_recovery(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(12, 20))
    q_in = rng.uniform(5e5, 8e6, n)
    t_cold = rng.uniform(285.0, 325.0, n)
    t_hot = _inject_nans(rng, t_cold + rng.uniform(90.0, 320.0, n), 0.14)
    for i in range(3):
        if np.isnan(t_hot[i]):
            t_hot[i] = t_cold[i] + 150.0
    df = pd.DataFrame({
        "unit_id": np.arange(n),
        "heat_input": np.round(q_in, 2),
        "t_hot": np.round(t_hot, 4),
        "t_cold": np.round(t_cold, 4),
    })
    return SciSpec(
        params={},
        tables={
            "exchangers": TableSpec(
                df=df,
                si_units={"heat_input": "J", "t_hot": "K", "t_cold": "K"},
                calibratable=["heat_input"],
                nullable_cols=["t_hot"],
                description="heat exchanger units (`unit_id`, `heat_input`, `t_hot`, `t_cold`)",
            )
        },
    )


def _solve_carnot_heat_recovery(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["exchangers"].dropna(subset=["heat_input", "t_hot", "t_cold"])
    eta = 1.0 - (df["t_cold"] / df["t_hot"])
    return float((df["heat_input"] * eta).sum()) / 1e6  # MJ


# =============================================================================
# 11. projectile_range_envelope (HELD-OUT, mech)
# =============================================================================
def _build_projectile_range_envelope(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(14, 22))
    v0 = rng.uniform(60.0, 220.0, n)
    theta = _inject_nans(rng, rng.uniform(0.30, 1.25, n), 0.14)  # rad
    for i in range(3):
        if np.isnan(theta[i]):
            theta[i] = 0.75
    df = pd.DataFrame({
        "trial_id": np.arange(n),
        "launch_speed": np.round(v0, 4),
        "launch_angle": np.round(theta, 6),
    })
    return SciSpec(
        params={},
        tables={
            "launches": TableSpec(
                df=df,
                si_units={"launch_speed": "m/s", "launch_angle": "rad"},
                calibratable=["launch_speed"],
                nullable_cols=["launch_angle"],
                description="ballistic trials (`trial_id`, `launch_speed`, `launch_angle`)",
            )
        },
    )


def _solve_projectile_range_envelope(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["launches"].dropna(subset=["launch_speed", "launch_angle"])
    r_m = (df["launch_speed"] ** 2) * np.sin(2.0 * df["launch_angle"]) / G_ACCEL
    return float(r_m.mean()) / 1e3  # km


# =============================================================================
# 12. concentration_mass_balance (HELD-OUT, thermo)
# =============================================================================
def _build_concentration_mass_balance(rng: np.random.Generator) -> SciSpec:
    n = int(rng.integers(12, 20))
    phase = list(rng.choice(["active", "standby"], n))
    for i in range(4):
        phase[i] = "active"
    vol_m3 = rng.uniform(0.5, 25.0, n)
    conc_g_l = _inject_nans(rng, rng.uniform(2.0, 60.0, n), 0.12)  # 1 g/L == 1 kg/m3
    for i in range(3):
        if np.isnan(conc_g_l[i]):
            conc_g_l[i] = 15.0
    df = pd.DataFrame({
        "tank_id": np.arange(n),
        "phase": phase,
        "volume": np.round(vol_m3, 5),
        "concentration": np.round(conc_g_l, 5),
    })
    return SciSpec(
        params={"phase": "active"},
        tables={
            "tanks": TableSpec(
                df=df,
                si_units={"volume": "m3", "concentration": "g/L"},
                calibratable=["concentration"],
                nullable_cols=["concentration"],
                description="chemical tanks (`tank_id`, `phase`, `volume`, `concentration`)",
            )
        },
    )


def _solve_concentration_mass_balance(tables: dict[str, pd.DataFrame], params: dict) -> float:
    df = tables["tanks"]
    sub = df[df["phase"] == params["phase"]].dropna(subset=["volume", "concentration"])
    # volume in m3, concentration in g/L == kg/m3 -> product is kg
    return float((sub["volume"] * sub["concentration"]).sum())


FAMILIES: list[SciFamily] = [
    SciFamily(
        "thermal_energy_budget", "thermo", False, "float",
        _build_thermal_energy_budget, _solve_thermal_energy_budget,
        lambda p: f"computes the total thermal energy change Q = sum(m * c * (temp_final - temp_init)) across all valid batches in zone `{p['zone']}`, in megajoules (MJ)",
    ),
    SciFamily(
        "hydrostatic_dam_force", "mech", False, "float",
        _build_hydrostatic_dam_force, _solve_hydrostatic_dam_force,
        lambda p: f"computes the total hydrostatic force sum((p_atm + fluid_density * 9.80665 * depth) * gate_area) on all valid `{p['status']}` gates, in kilonewtons (kN)",
    ),
    SciFamily(
        "beam_bending_stress", "mech", False, "float",
        _build_beam_bending_stress, _solve_beam_bending_stress,
        lambda p: "joins `beams.csv` with `sections.csv` on `section_code` and computes the maximum mid-span bending stress sigma = (point_load * span_length / 4) * half_depth / inertia across valid beams, in megapascals (MPa)",
    ),
    SciFamily(
        "trapezoidal_power_energy", "timeseries", False, "float",
        _build_trapezoidal_power_energy, _solve_trapezoidal_power_energy,
        lambda p: "integrates valid power readings over `elapsed_time` using the trapezoidal rule to compute total energy, in kilowatt-hours (kWh)",
    ),
    SciFamily(
        "ideal_gas_moles_filtered", "thermo", False, "float",
        _build_ideal_gas_moles_filtered, _solve_ideal_gas_moles_filtered,
        lambda p: f"computes the total moles of gas n = sum(P * V / (8.314462618 * T)) across all valid `{p['gas_type']}` vessels (using absolute temperature), in moles (mol)",
    ),
    SciFamily(
        "kinetic_braking_work", "mech", False, "float",
        _build_kinetic_braking_work, _solve_kinetic_braking_work,
        lambda p: "joins `braking_events.csv` with `vehicles.csv` on `model_code` and computes total kinetic energy dissipated sum(0.5 * m * (v_initial^2 - v_final^2)) across valid events, in megajoules (MJ)",
    ),
    SciFamily(
        "pipe_darcy_headloss", "mech", False, "float",
        _build_pipe_darcy_headloss, _solve_pipe_darcy_headloss,
        lambda p: f"computes total Darcy-Weisbach head loss sum(f * (L / D) * v^2 / (2 * 9.80665)) across all valid `{p['line_group']}` pipe segments, in metres (m)",
    ),
    SciFamily(
        "control_weighted_sensor_mean", "control", False, "float",
        _build_control_weighted_sensor_mean, _solve_control_weighted_sensor_mean,
        lambda p: "computes the mean total `alert_count` per site across ALL sites listed in `sites.csv` (including sites with zero alerts)",
    ),
    SciFamily(
        "solar_panel_yield_kwh", "timeseries", True, "float",
        _build_solar_panel_yield_kwh, _solve_solar_panel_yield_kwh,
        lambda p: "integrates electrical power output (irradiance * panel_area * efficiency) over `elapsed_time` using the trapezoidal rule across valid readings, in kilowatt-hours (kWh)",
    ),
    SciFamily(
        "carnot_heat_recovery", "thermo", True, "float",
        _build_carnot_heat_recovery, _solve_carnot_heat_recovery,
        lambda p: "computes total Carnot maximum recoverable work sum(heat_input * (1 - t_cold / t_hot)) across all valid exchanger units (using absolute temperatures), in megajoules (MJ)",
    ),
    SciFamily(
        "projectile_range_envelope", "mech", True, "float",
        _build_projectile_range_envelope, _solve_projectile_range_envelope,
        lambda p: "computes the mean drag-free projectile range (launch_speed^2 * sin(2 * launch_angle) / 9.80665) across all valid trials, in kilometres (km)",
    ),
    SciFamily(
        "concentration_mass_balance", "thermo", True, "float",
        _build_concentration_mass_balance, _solve_concentration_mass_balance,
        lambda p: f"computes total dissolved solute mass sum(volume * concentration) across all valid `{p['phase']}` tanks, in kilograms (kg)",
    ),
]

BY_NAME: dict[str, SciFamily] = {f.name: f for f in FAMILIES}

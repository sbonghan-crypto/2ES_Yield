from pathlib import Path
import numpy as np
import pandas as pd
from utils import (
    conservative_hz_au,
    kepler_period_days,
    rv_k50_mps,
    rv_mass_limit_mearth,
    sma_au_from_period_days,
)

SOURCE_ID_COL = "source_id"
TEFF_COL = "T_eff [K]"
LUMINOSITY_COL = "Luminosity [L_Sun]"
MASS_COL = "Mass [M_Sun]"
YEARLY_OBSERVABLE_NIGHTS_COL = "Yearly_Observable_nights"
SIGMA_RV_TOTAL_COL = "σ_RV,total [m/s]"

N_PERIOD_GRID = 500
SURVEY_YEARS = 5
OBSERVING_RATE = 0.8
DETECTION_THRESHOLD_PARAMETER = 6.0

DETECTION_LIMIT_DTYPE = np.dtype(
    [
        ("source_id", "U256"),
        ("period_days", "f8"),
        ("sma_au", "f8"),
        ("k_sigma_rv_total_mps", "f8"),
        ("mass_limit_mearth", "f8"),
    ]
)

DETECTED_PLANET_DTYPE = np.dtype(
    [
        ("source_id", "U256"),
        ("mc_run", "i8"),
        ("radius_rearth", "f8"),
        ("mass_mearth", "f8"),
        ("sma_au", "f8"),
        ("period_days", "f8"),
        ("nearest_limit_period_days", "f8"),
        ("mass_limit_mearth", "f8"),
        ("detected", "?"),
    ]
)


def get_mass_limit(
    cat_path: str | Path,
    output_path: str | Path,
    n_period_grid: int = N_PERIOD_GRID,
    survey_years: float = SURVEY_YEARS,
    observing_rate: float = OBSERVING_RATE,
    detection_threshold_parameter: float = DETECTION_THRESHOLD_PARAMETER,
) -> Path:
    
    cat_path = Path(cat_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(cat_path)

    for _, star in cat.iterrows():
        source_id = str(star[SOURCE_ID_COL])
        teff_k = float(star[TEFF_COL])
        luminosity_lsun = float(star[LUMINOSITY_COL])
        mass_msun = float(star[MASS_COL])
        yearly_observable_nights = float(star[YEARLY_OBSERVABLE_NIGHTS_COL])
        sigma_rv_total_mps = float(star[SIGMA_RV_TOTAL_COL])
        inner_hz_au, outer_hz_au = conservative_hz_au(teff_k, luminosity_lsun)
        period_min_days, period_max_days = kepler_period_days(
            np.array([inner_hz_au, outer_hz_au]), mass_msun
        )
        period_days = np.linspace(period_min_days, period_max_days, n_period_grid)
        sma_au = sma_au_from_period_days(period_days, mass_msun)
        n_obs = observing_rate * survey_years * yearly_observable_nights
        k_sigma_rv_total_mps = rv_k50_mps(
            period_days,
            sigma_rv_total_mps,
            n_obs,
            t_span_years=survey_years,
            detection_threshold_parameter=detection_threshold_parameter,
        )
        mass_limit_mearth = rv_mass_limit_mearth(
            k_sigma_rv_total_mps, sma_au, mass_msun
        )

        limits = np.empty(n_period_grid, dtype=DETECTION_LIMIT_DTYPE)
        limits[SOURCE_ID_COL] = source_id
        limits["period_days"] = period_days
        limits["sma_au"] = sma_au
        limits["k_sigma_rv_total_mps"] = k_sigma_rv_total_mps
        limits["mass_limit_mearth"] = mass_limit_mearth
        np.save(out_dir / f"{source_id}.npy", limits)

    return out_dir


def run_yield(
    cat_path: str | Path,
    gen_planet_path: str | Path,
    det_limit_path: str | Path,
    output_path: str | Path,
    n_mc: int,
) -> Path:
    
    cat = pd.read_csv(cat_path)

    gen_planet_path = Path(gen_planet_path)
    det_limit_path = Path(det_limit_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source_id in cat[SOURCE_ID_COL].astype(str):
        planet_file = gen_planet_path / f"{source_id}.npy"
        limit_file = det_limit_path / f"{source_id}.npy"

        planets = np.load(planet_file)
        limits = np.load(limit_file)

        detected_planets = np.empty(len(planets), dtype=DETECTED_PLANET_DTYPE)
        if len(planets) > 0:
            limit_periods = limits["period_days"]
            limit_masses = limits["mass_limit_mearth"]

            nearest_indices = np.abs(
                planets["period_days"][:, None] - limit_periods[None, :]
            ).argmin(axis=1)

            nearest_limit_periods = limit_periods[nearest_indices]
            nearest_mass_limits = limit_masses[nearest_indices]
            detected = planets["mass_mearth"] >= nearest_mass_limits

            detected_planets[SOURCE_ID_COL] = planets[SOURCE_ID_COL]
            detected_planets["mc_run"] = planets["mc_run"]
            detected_planets["radius_rearth"] = planets["radius_rearth"]
            detected_planets["mass_mearth"] = planets["mass_mearth"]
            detected_planets["sma_au"] = planets["sma_au"]
            detected_planets["period_days"] = planets["period_days"]
            detected_planets["nearest_limit_period_days"] = nearest_limit_periods
            detected_planets["mass_limit_mearth"] = nearest_mass_limits
            detected_planets["detected"] = detected

        np.save(out_dir / f"{source_id}.npy", detected_planets)
    return out_dir

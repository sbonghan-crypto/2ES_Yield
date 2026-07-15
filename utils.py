import numpy as np

# Coefficients

KOPPARAPU_2013 = {
    "moist_greenhouse": {
        "S_eff_sun": 1.0140,
        "a": 8.1774e-5,
        "b": 1.7063e-9,
        "c": -4.3241e-12,
        "d": -6.6462e-16,
    },
    "maximum_greenhouse": {
        "S_eff_sun": 0.3438,
        "a": 5.8942e-5,
        "b": 1.6558e-9,
        "c": -3.0045e-12,
        "d": -5.2983e-16,
    },
}

# Constants

AU = 1.49597e11  # meters
G = 6.67430e-11  # m^3 kg^-1 s
M_SUN = 1.9885e30  # kg
M_EARTH = 5.9722e24  # kg

# Utility functions for habitable zone calculations

# Kopparapu et al. (2013)

def seff(teff_k: float, hz_zone: str) -> float:
    coeff = KOPPARAPU_2013[hz_zone]
    t_star = teff_k - 5780.0
    return (
        coeff["S_eff_sun"]
        + coeff["a"] * t_star
        + coeff["b"] * t_star**2
        + coeff["c"] * t_star**3
        + coeff["d"] * t_star**4
    )


def hz_distance_au(teff_k: float, luminosity_lsun: float, hz_zone: str) -> float:
    return float(np.sqrt(luminosity_lsun / seff(teff_k, hz_zone)))


def conservative_hz_au(teff_k: float, luminosity_lsun: float) -> tuple[float, float]:
    inner = hz_distance_au(teff_k, luminosity_lsun, "moist_greenhouse")
    outer = hz_distance_au(teff_k, luminosity_lsun, "maximum_greenhouse")
    return inner, outer

# Period and semi-major axis conversions

def kepler_period_days(sma_au: np.ndarray, stellar_mass_msun: float) -> np.ndarray:
    return 365 * np.sqrt(np.array(sma_au) ** 3 / stellar_mass_msun)


def sma_au_from_period_days(period_days: np.ndarray, stellar_mass_msun: float) -> np.ndarray:
    period_yr = np.array(period_days) / 365
    return (period_yr**2 * stellar_mass_msun) ** (1/3)

# Müller et al. (2024)

def mass_from_radius_mearth(radius_rearth: np.ndarray) -> np.ndarray:
    # For M_earth < 4.37
    radius_rearth = np.array(radius_rearth)
    mass_mearth = (radius_rearth / 1.02) ** (1.0 / 0.27)
    return mass_mearth

# Detection limit calculations

def rv_k50_mps(
    period_days: np.ndarray,
    sigma_rv_mps: float,
    n_obs: float,
    t_span_years: float,
    detection_threshold_parameter: float,
) -> np.ndarray:
    period_yr = np.array(period_days) / 365
    tau = period_yr / t_span_years
    return sigma_rv_mps * detection_threshold_parameter / np.sqrt(n_obs) * np.sqrt(1.0 + (10.0 ** (tau - 1.5)) ** 2)


def rv_mass_limit_mearth(
    k_mps: np.ndarray,
    sma_au: np.ndarray,
    stellar_mass_msun: float,
) -> np.ndarray:
    return np.sqrt(AU*M_SUN/G)/M_EARTH* np.array(k_mps) * np.sqrt(
        np.array(sma_au) * stellar_mass_msun
    )

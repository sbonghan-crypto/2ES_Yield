from pathlib import Path
import numpy as np
import pandas as pd
from utils import (
    conservative_hz_au, 
    kepler_period_days,
    mass_from_radius_mearth
    )


SOURCE_ID_COL = "source_id"
TEFF_COL = "T_eff [K]"
LUMINOSITY_COL = "Luminosity [L_Sun]"
MASS_COL = "Mass [M_Sun]"

OCCURRENCE_RATE = 0.099
MIN_RADIUS_REARTH = 0.7
MAX_RADIUS_REARTH = 1.5

PLANET_DTYPE = np.dtype(
    [
        ("source_id", "U256"),
        ("mc_run", "i8"),
        ("radius_rearth", "f8"),
        ("mass_mearth", "f8"),
        ("sma_au", "f8"),
        ("period_days", "f8"),
    ]
)


def run(
    cat_path: str | Path,
    output_path: str | Path,
    n_mc: int = 1000
) -> Path:
    
    cat_path = Path(cat_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(cat_path)

    rng = np.random.default_rng(11)
    hz_rows = []
    summary_rows = []

    for _, star in cat.iterrows():
        source_id = str(star[SOURCE_ID_COL])
        teff_k = float(star[TEFF_COL])
        luminosity_lsun = float(star[LUMINOSITY_COL])
        mass_msun = float(star[MASS_COL])

        inner_hz_au, outer_hz_au = conservative_hz_au(teff_k, luminosity_lsun)
        inner_hz_period_days, outer_hz_period_days = kepler_period_days(inner_hz_au, mass_msun), kepler_period_days(outer_hz_au, mass_msun)

        hz_rows.append(
            {
                SOURCE_ID_COL: source_id,
                "inner_hz_au": inner_hz_au,
                "outer_hz_au": outer_hz_au,
                "inner_hz_period_days": inner_hz_period_days,
                "outer_hz_period_days": outer_hz_period_days,
            }
        )

        planet_counts = rng.poisson(OCCURRENCE_RATE, size=n_mc)
        total_planets = int(planet_counts.sum())
        planets = np.empty(total_planets, dtype=PLANET_DTYPE)

        if total_planets > 0:
            mc_run = np.repeat(np.arange(n_mc, dtype=np.int64), planet_counts)
            radius_rearth = rng.uniform(
                MIN_RADIUS_REARTH, MAX_RADIUS_REARTH, size=total_planets
            )
            mass_mearth = mass_from_radius_mearth(radius_rearth)
            sma_au = rng.uniform(inner_hz_au, outer_hz_au, size=total_planets)
            period_days = kepler_period_days(sma_au, mass_msun)

            planets[SOURCE_ID_COL] = source_id
            planets["mc_run"] = mc_run
            planets["radius_rearth"] = radius_rearth
            planets["mass_mearth"] = mass_mearth
            planets["sma_au"] = sma_au
            planets["period_days"] = period_days

        np.save(out_dir / f"{source_id}.npy", planets)

    hz_catalogue = pd.DataFrame(hz_rows)
    hz_catalogue.to_csv(out_dir / "conservative_hz_catalogue.csv", index=False)
    return out_dir

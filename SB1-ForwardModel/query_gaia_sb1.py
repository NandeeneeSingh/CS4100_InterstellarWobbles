from astroquery.gaia import Gaia
from pathlib import Path

RAW_DIR = Path("data/sb1-model/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RAW_DIR / "gaia_sb1_raw.csv"

QUERY = """
SELECT TOP 5000
    nss.source_id,
    gs.phot_g_mean_mag,
    nss.solution_id,
    nss.nss_solution_type,
    nss.ra,
    nss.dec,
    nss.parallax,
    nss.period,
    nss.period_error,
    nss.t_periastron,
    nss.t_periastron_error,
    nss.eccentricity,
    nss.eccentricity_error,
    nss.arg_periastron,
    nss.arg_periastron_error,
    nss.center_of_mass_velocity,
    nss.center_of_mass_velocity_error,
    nss.semi_amplitude_primary,
    nss.semi_amplitude_primary_error,
    nss.inclination,
    nss.inclination_error,
    nss.rv_n_obs_primary,
    nss.rv_n_good_obs_primary,
    nss.efficiency,
    nss.significance,
    nss.goodness_of_fit,
    nss.conf_spectro_period,
    nss.flags
FROM gaiadr3.nss_two_body_orbit AS nss
JOIN gaiadr3.gaia_source AS gs ON nss.source_id = gs.source_id
WHERE nss.nss_solution_type = 'SB1'
  AND nss.period BETWEEN 2 AND 800
"""

def main():
    print("Launching Gaia query...")
    job = Gaia.launch_job(QUERY)
    results = job.get_results()

    print(f"Rows returned: {len(results)}")

    results.write(OUTPUT_CSV, format="csv", overwrite=True)
    print(f"Saved raw data to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
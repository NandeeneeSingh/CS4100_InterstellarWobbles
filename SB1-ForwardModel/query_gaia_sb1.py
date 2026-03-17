from astroquery.gaia import Gaia
from pathlib import Path

RAW_DIR = Path("data/sb1-model/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RAW_DIR / "gaia_sb1_raw.csv"

QUERY = """
SELECT TOP 5000
    source_id,
    solution_id,
    nss_solution_type,
    ra,
    dec,
    parallax,
    period,
    period_error,
    t_periastron,
    t_periastron_error,
    eccentricity,
    eccentricity_error,
    arg_periastron,
    arg_periastron_error,
    center_of_mass_velocity,
    center_of_mass_velocity_error,
    semi_amplitude_primary,
    semi_amplitude_primary_error,
    inclination,
    inclination_error,
    rv_n_obs_primary,
    rv_n_good_obs_primary,
    efficiency,
    significance,
    goodness_of_fit,
    conf_spectro_period,
    flags
FROM gaiadr3.nss_two_body_orbit
WHERE nss_solution_type = 'SB1'
  AND rv_n_good_obs_primary >= 20
  AND efficiency >= 0.2
  AND significance >= 10
  AND period BETWEEN 2 AND 800
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
import argparse
import numpy as np
import pandas as pd
import os

PREFACTOR = 1.036e-7       # converts P[days], K[km/s] -> solar masses
SIGNIFICANCE_THRESHOLD = 5.0
OUTPUT_DIR = 'data/04_categorized'


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def mass_function_sb1(P, e, K1):
    """f(M) = 1.036e-7 * P * K1^3 * (1 - e^2)^1.5  [solar masses]"""
    return PREFACTOR * P * K1**3 * (1 - e**2)**1.5


def solve_m2_sb1_batch(fm, m1, sin3i, max_iter=100, tol=1e-8):
    """
    Vectorized Newton solver for M2 from the SB1 mass function.

    Solves:  M2^3 * sin^3(i) - f(M) * (M1 + M2)^2 = 0

    Initial guess uses the M2 << M1 approximation:
        M2_0 ~ (f(M) * M1^2 / sin^3(i))^(1/3)
    """
    # Guard against non-physical inputs
    bad = (fm <= 0) | (m1 <= 0) | (sin3i <= 0)
    fm = np.where(bad, np.nan, fm)

    m2 = np.where(
        np.isfinite(fm),
        np.cbrt(fm * m1**2 / np.where(sin3i > 0, sin3i, 1.0)),
        1.0
    )
    m2 = np.clip(m2, 1e-9, None)

    for _ in range(max_iter):
        f  = m2**3 * sin3i - fm * (m1 + m2)**2
        fp = 3 * m2**2 * sin3i - 2 * fm * (m1 + m2)
        fp = np.where(np.abs(fp) < 1e-30, 1e-30, fp)
        dm2 = f / fp
        m2 = np.clip(m2 - dm2, 1e-9, None)
        if np.nanmax(np.abs(dm2)) < tol:
            break

    # Mark non-converged draws as invalid
    residual = np.abs(m2**3 * sin3i - fm * (m1 + m2)**2)
    norm = np.abs(fm * (m1 + m2)**2) + 1e-30
    m2[residual / norm > 1e-4] = np.nan
    m2[bad] = np.nan
    return m2


def m2_sin3i_sb2(P, e, K1, K2):
    """M2*sin^3(i) = 1.036e-7 * P * (1-e^2)^1.5 * (K1+K2)^2 * K1"""
    return PREFACTOR * P * (1 - e**2)**1.5 * (K1 + K2)**2 * K1


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _artifact_result():
    return {
        'p_substellar': 0.0,
        'p_degenerate_compact': 0.0,
        'p_high_mass_dark_remnant': 0.0,
        'p_systemic_artifact': 1.0,
        'most_likely_category': 'Systemic Artifact',
        'category_confidence': 1.0,
        'm2_p16': np.nan,
        'm2_p50': np.nan,
        'm2_p84': np.nan,
        'invalid_draw_fraction': 1.0,
    }


def _build_result(m2_samples, n_iter):
    valid_mask = np.isfinite(m2_samples) & (m2_samples > 0)
    valid = m2_samples[valid_mask]
    invalid_frac = 1.0 - valid_mask.sum() / n_iter

    p_sub = np.sum(valid < 0.08) / n_iter
    p_deg = np.sum((valid >= 0.08) & (valid <= 3.0)) / n_iter
    p_rem = np.sum(valid > 3.0) / n_iter
    p_art = max(0.0, 1.0 - p_sub - p_deg - p_rem)

    probs = {
        'Substellar Perturbator':    p_sub,
        'Degenerate Compact Object': p_deg,
        'High-Mass Dark Remnant':    p_rem,
        'Systemic Artifact':         p_art,
    }
    most_likely = max(probs, key=probs.get)

    if len(valid) > 0:
        m2_p16, m2_p50, m2_p84 = np.percentile(valid, [16, 50, 84])
    else:
        m2_p16 = m2_p50 = m2_p84 = np.nan

    return {
        'p_substellar':            p_sub,
        'p_degenerate_compact':    p_deg,
        'p_high_mass_dark_remnant': p_rem,
        'p_systemic_artifact':     p_art,
        'most_likely_category':    most_likely,
        'category_confidence':     probs[most_likely],
        'm2_p16':                  m2_p16,
        'm2_p50':                  m2_p50,
        'm2_p84':                  m2_p84,
        'invalid_draw_fraction':   invalid_frac,
    }


def _sample_with_fallback(row, col, fallback_frac=0.01):
    """Return (value, error) for a column, falling back to fractional error."""
    err_col = col + '_error'
    val = row[col]
    if err_col in row.index and pd.notna(row[err_col]):
        return val, row[err_col]
    return val, abs(val) * fallback_frac


# ---------------------------------------------------------------------------
# Per-source-type MC runners
# ---------------------------------------------------------------------------

def run_sb1(df, n_iter):
    results = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"  SB1: {idx + 1}/{total}")

        sig = row.get('significance') if hasattr(row, 'get') else row['significance'] if 'significance' in row.index else np.nan
        if pd.notna(sig) and sig < SIGNIFICANCE_THRESHOLD:
            results.append(_artifact_result())
            continue

        # Isotropic inclination sampling
        cos_i  = np.random.uniform(0, 1, n_iter)
        sin3i  = (np.sqrt(1 - cos_i**2))**3

        # M1: 10% fractional uncertainty (no dedicated error column)
        m1_val = row['m1_solar_m']
        m1 = np.clip(np.random.normal(m1_val, m1_val * 0.1, n_iter), 1e-6, None)

        P_val,  P_err  = _sample_with_fallback(row, 'period')
        e_val,  e_err  = _sample_with_fallback(row, 'eccentricity', fallback_frac=0.05)
        K1_val, K1_err = _sample_with_fallback(row, 'semi_amplitude_primary')

        P  = np.clip(np.random.normal(P_val,  P_err,  n_iter), 1e-3, None)
        e  = np.clip(np.random.normal(e_val,  e_err,  n_iter), 0.0,  0.999)
        K1 = np.clip(np.random.normal(K1_val, K1_err, n_iter), 1e-6, None)

        fm = mass_function_sb1(P, e, K1)
        m2 = solve_m2_sb1_batch(fm, m1, sin3i)
        results.append(_build_result(m2, n_iter))

    return results


def run_sb2(df, n_iter):
    results = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"  SB2: {idx + 1}/{total}")

        sig = row.get('significance') if hasattr(row, 'get') else row['significance'] if 'significance' in row.index else np.nan
        if pd.notna(sig) and sig < SIGNIFICANCE_THRESHOLD:
            results.append(_artifact_result())
            continue

        # Isotropic inclination sampling — guard against sin(i)=0
        cos_i = np.random.uniform(0, 1, n_iter)
        sin3i = np.clip((np.sqrt(1 - cos_i**2))**3, 1e-9, None)

        P_val,  P_err  = _sample_with_fallback(row, 'period')
        e_val,  e_err  = _sample_with_fallback(row, 'eccentricity', fallback_frac=0.05)
        K1_val, K1_err = _sample_with_fallback(row, 'semi_amplitude_primary')
        K2_val, K2_err = _sample_with_fallback(row, 'semi_amplitude_secondary')

        P  = np.clip(np.random.normal(P_val,  P_err,  n_iter), 1e-3, None)
        e  = np.clip(np.random.normal(e_val,  e_err,  n_iter), 0.0,  0.999)
        K1 = np.clip(np.random.normal(K1_val, K1_err, n_iter), 1e-6, None)
        K2 = np.clip(np.random.normal(K2_val, K2_err, n_iter), 1e-6, None)

        m2_s3i = m2_sin3i_sb2(P, e, K1, K2)
        m2 = m2_s3i / sin3i

        result = _build_result(m2, n_iter)
        result['p_systemic_artifact'] = 0.0
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Spectroscopic MC companion classifier')
    parser.add_argument('--input',       required=True,  help='Path to input CSV')
    parser.add_argument('--source-type', required=True,  choices=['sb1', 'sb2'],
                        help='Spectroscopic binary type')
    parser.add_argument('--iterations',  type=int, default=10000,
                        help='MC draws per source (default: 10000)')
    parser.add_argument('--output',      required=True,  help='Path to output CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Running {args.iterations} MC iterations per source ({args.source_type.upper()})...")

    if args.source_type == 'sb1':
        result_rows = run_sb1(df, args.iterations)
    else:
        result_rows = run_sb2(df, args.iterations)

    results_df = pd.DataFrame(result_rows, index=df.index)
    out_df = pd.concat([df, results_df], axis=1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows to {args.output}")
    all_categories = [
        'Substellar Perturbator',
        'Degenerate Compact Object',
        'High-Mass Dark Remnant',
        'Systemic Artifact',
    ]
    counts = out_df['most_likely_category'].value_counts()
    print("\nCategory breakdown:")
    for cat in all_categories:
        print(f"  {cat}: {counts.get(cat, 0)}")


if __name__ == '__main__':
    main()

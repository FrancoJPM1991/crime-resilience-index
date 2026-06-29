import numpy as np
import pandas as pd
from src.config import DATA_INTERIM


def _assign_state(crime_rate: float, regime_entry: dict) -> int:
    """
    Assign a quantile state to a municipality based on its crime rate
    and the bin edges stored in regime_meta.

    Uses bin_lower and bin_upper dicts keyed by state integer.
    The lowest state's bin_lower is inclusive; all bin_upper are inclusive.
    Falls back to the closest state if the rate sits exactly on a boundary.
    """
    bin_lower = regime_entry['bin_lower']
    bin_upper = regime_entry['bin_upper']
    n_states  = regime_entry['n_states']

    for state in range(1, n_states + 1):
        lo = bin_lower[state]
        hi = bin_upper[state]
        if lo <= crime_rate <= hi:
            return state

    # Fallback: clamp to nearest state (handles floating-point edge cases)
    if crime_rate < bin_lower[1]:
        return 1
    return n_states


def _compute_mobility(P: np.ndarray, state: int) -> tuple[float, float, float]:
    """
    Compute downward mobility D, upward pressure U, and persistence π
    for a given state (1-indexed) from transition matrix P.

    Returns
    -------
    D   : float  probability mass flowing to lower states
    U   : float  probability mass flowing to higher states
    pi  : float  self-transition probability (diagonal)
    """
    idx = state - 1  # convert to 0-indexed row
    row = P[idx]

    D  = float(row[:idx].sum())          # states below current
    U  = float(row[idx + 1:].sum())      # states above current
    pi = float(row[idx])                 # diagonal

    return D, U, pi


def _compute_steady_state_gap(
    state: int,
    regime_entry: dict
) -> float:
    """
    Compute the steady-state gap G for a municipality.

    G = (ss_expected_rate - state_mean) / (max_state_mean - min_state_mean)

    Positive G: municipality is below the regime attractor → vulnerability signal.
    Negative G: municipality is above the regime attractor → resilience signal.

    Returns 0.0 for single-state regimes (regime 1) to avoid division by zero.
    """
    state_means    = regime_entry['state_means']
    ss_rate        = regime_entry['ss_expected_rate']
    means_vals     = list(state_means.values())

    denom = max(means_vals) - min(means_vals)
    if denom == 0:
        return 0.0

    G = (ss_rate - state_means[state]) / denom
    return float(G)


def compute_cri_components(regime_meta: dict, year: int = 2024) -> pd.DataFrame:
    """
    Compute raw CRI components for all municipalities in a given year.

    For each municipality, assigns its quantile state within its 2024 regime,
    then computes TRS, SCS, and PP from the transition matrix dynamics.

    Parameters
    ----------
    regime_meta : dict
        Output of build_regime_metadata() from cri_weights.py.
    year : int
        Base year for state assignment (default 2024).

    Returns
    -------
    pd.DataFrame with columns:
        CVEGEO, regime, current_state, crime_rate_2024,
        D, U, pi, G, TRS, SCS, PP
    """
    # Load and filter to base year
    df = pd.read_csv(f"{DATA_INTERIM}/crime_rates_regimes.csv", encoding="latin-1")
    df['CVEGEO'] = df['CVEGEO'].astype(str).str.zfill(5)
    df = df[df['year'] == year].copy()

    print(f"\n*** Computing CRI components for year {year} ***")
    print(f"  Municipalities loaded: {len(df)}")

    n_before = len(df)
    df = df[df['regimes'].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  WARNING: {n_dropped} municipalities dropped due to NaN regime assignment.")
    print(f"  Municipalities after NaN filter: {len(df)}")

    records = []

    for _, row in df.iterrows():
        cvegeo      = row['CVEGEO']
        crime_rate  = row['crime_rate']
        regime_int  = int(row['regimes'])
        regime_name = f"df_regime{regime_int}"

        if regime_name not in regime_meta:
            print(f"  WARNING: {cvegeo} assigned to {regime_name} not found in regime_meta — skipping.")
            continue

        entry = regime_meta[regime_name]
        P     = entry['transition_matrix']
        omega = entry['severity_weight']

        # --- State assignment ---
        state = _assign_state(crime_rate, entry)

        # --- Mobility components ---
        D, U, pi = _compute_mobility(P, state)

        # --- Steady-state gap ---
        G = _compute_steady_state_gap(state, entry)

        # --- TRS: net downward mobility scaled by regime severity ---
        TRS = (D - U) * omega

        # --- SCS: negative gap (above attractor = resilient) ---
        SCS = -G

        # --- PP: reward low-state persistence, penalize high-state persistence ---
        # Threshold: states 1-2 are low, states 3-5 are high
        # For single-state regimes (regime 1), state=1 always → PP = +pi
        low_state_threshold = 2
        PP = pi if state <= low_state_threshold else -pi

        records.append({
            'CVEGEO':          cvegeo,
            'regime':          regime_name,
            'current_state':   state,
            'crime_rate_2024': round(crime_rate, 4),
            'D':               round(D,   4),
            'U':               round(U,   4),
            'pi':              round(pi,  4),
            'G':               round(G,   4),
            'TRS':             round(TRS, 4),
            'SCS':             round(SCS, 4),
            'PP':              round(PP,  4),
        })

    components_df = pd.DataFrame(records)

    # --- Diagnostic summary by regime ---
    print("\n  Component means by regime:")
    summary = (
        components_df
        .groupby('regime')[['TRS', 'SCS', 'PP']]
        .mean()
        .round(4)
    )
    print(summary)
    print(f"\n  Total municipalities processed: {len(components_df)}")
    print("*** CRI components computed successfully ***\n")

    return components_df
import numpy as np
import pandas as pd
from src.config import DATA_INTERIM


REGIME_NAMES = [f"df_regime{i}" for i in range(1, 6)]


def _load_transition_matrix(regime_name: str) -> np.ndarray:
    """Load transition matrix CSV for a given regime and return as numpy array."""
    path = f"{DATA_INTERIM}/T_{regime_name}.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.values.astype(float)


def _load_state_meta(regime_name: str) -> pd.DataFrame:
    """Load state metadata CSV for a given regime."""
    path = f"{DATA_INTERIM}/state_meta_{regime_name}.csv"
    df = pd.read_csv(path)
    df['state'] = df['state'].astype(int)
    return df.set_index('state')


def _solve_steady_state(P: np.ndarray) -> np.ndarray:
    """
    Solve for the steady-state distribution μ such that μP = μ.
    Uses the left eigenvector corresponding to eigenvalue 1
    of the transposed transition matrix.
    Returns a normalized probability vector summing to 1.
    """
    n = P.shape[0]

    if n == 1:
        return np.array([1.0])

    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    steady = np.real(eigenvectors[:, idx])

    # Normalize to sum to 1 (eigenvectors are not probability vectors by default)
    steady = steady / steady.sum()

    return steady


def build_regime_metadata(n_regimes: int = 5) -> dict:
    """
    Load transition matrices and state metadata for all regimes,
    compute steady-state distributions and severity weights,
    and return a structured dict for use by cri_components.py.

    Parameters
    ----------
    n_regimes : int
        Number of regimes in the pipeline (default 5, after thin-regime merging).

    Returns
    -------
    regime_meta : dict
        Keys are regime name strings (e.g. 'df_regime1').
        Each value is a dict with:
            - transition_matrix : np.ndarray  (n_states x n_states)
            - state_means       : dict        {state_int: mean_crime_rate}
            - bin_lower         : dict        {state_int: lower_bin_edge}
            - bin_upper         : dict        {state_int: upper_bin_edge}
            - steady_state      : np.ndarray  (n_states,) sums to 1
            - ss_expected_rate  : float       steady-state weighted mean crime rate
            - severity_weight   : float       ω(r), computed after all regimes loaded
            - n_states          : int
    """
    regime_names = REGIME_NAMES[:n_regimes]
    regime_meta = {}

    print("\n*** Building CRI regime metadata ***")

    for name in regime_names:
        P = _load_transition_matrix(name)
        meta_df = _load_state_meta(name)

        n_states = P.shape[0]
        state_means = meta_df['mean'].to_dict()
        bin_lower = meta_df['bin_lower'].to_dict()
        bin_upper = meta_df['bin_upper'].to_dict()

        steady = _solve_steady_state(P)
        means_vec = np.array([state_means[s] for s in sorted(state_means.keys())])
        ss_expected_rate = float(np.dot(steady, means_vec))

        regime_meta[name] = {
            'transition_matrix': P,
            'state_means':       state_means,
            'bin_lower':         bin_lower,
            'bin_upper':         bin_upper,
            'steady_state':      steady,
            'ss_expected_rate':  ss_expected_rate,
            'severity_weight':   None,   # filled in after all regimes loaded
            'n_states':          n_states
        }

        print(f"  {name}: {n_states} states | "
              f"ss_expected_rate={ss_expected_rate:.3f} | "
              f"steady_state={np.round(steady, 3)}")

    # --- Compute severity weights ω(r) across all regimes ---
    # Uses log-normalized steady-state expected rate so that
    # low-violence regimes receive weight close to 1 and
    # high-violence regimes receive weight close to 0.
    ss_rates = np.array([regime_meta[n]['ss_expected_rate'] for n in regime_names])
    log_rates = np.log(ss_rates + 1)   # +1 to handle near-zero rates safely
    log_max = log_rates.max()

    for name, log_rate in zip(regime_names, log_rates):
        omega = float(1.0 - (log_rate / log_max))
        regime_meta[name]['severity_weight'] = omega
        print(f"  {name}: severity_weight ω = {omega:.4f}")

    print("*** Regime metadata built successfully ***\n")

    return regime_meta
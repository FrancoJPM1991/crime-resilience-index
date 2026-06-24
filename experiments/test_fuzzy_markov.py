import itertools
import warnings
import pandas as pd
import numpy as np
from src.models.regime import regime_construction
from src.models.quantile import assign_quantiles
from src.models.transition import transition_matrix
from src.models.thin_regimes import merge_thin_regimes

# ---------------------------------------------------------------------------
# Grid definition — edit ranges here
# ---------------------------------------------------------------------------
PARAM_GRID = {
    'N_REGIMES':   [3, 4, 5, 6, 7],
    'N_QUANTILES': [3, 4, 5, 6],
    'MIN_MUNI':    [20, 30, 40],
}

YEAR_BASE   = 2024
YEAR_TARGET = 2025
N_STEPS     = 1


# ---------------------------------------------------------------------------
# Single-run forecast (mirrors markov_prediction but returns raw df)
# ---------------------------------------------------------------------------
def _run_forecast(df_input, N_REGIMES, N_QUANTILES, MIN_MUNI):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        regime_dfs        = regime_construction(df_input, N_REGIMES)
        merged_regimes    = merge_thin_regimes(regime_dfs, MIN_MUNI)
        quantile_dfs      = assign_quantiles(merged_regimes, N_QUANTILES)
        transition_results = transition_matrix(quantile_dfs)

    records = []
    for reg_name, df in quantile_dfs.items():
        T_data     = transition_results[reg_name]
        T          = T_data['T'].values
        states     = T_data['states']
        state_means = T_data['state_means']

        base = (
            df[df['year'] == YEAR_BASE][['CVEGEO', 'state', 'membership']]
            .drop_duplicates('CVEGEO')
        )

        for _, row in base.iterrows():
            mun        = row['CVEGEO']
            membership = row['membership']

            pi = np.zeros(len(states))
            if isinstance(membership, dict):
                for s, mu in membership.items():
                    if s in states:
                        pi[states.index(s)] = mu
            else:
                s0 = int(row['state'])
                pi[states.index(s0)] = 1.0

            pi_sum = pi.sum()
            if pi_sum > 0:
                pi /= pi_sum

            T_n   = np.linalg.matrix_power(T, N_STEPS)
            pi_n  = pi @ T_n
            crisp = float(np.dot(pi_n, [state_means[s] for s in states]))

            records.append({'CVEGEO': mun, 'markov_fuzzy': round(crisp, 2)})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# RMSE helper
# ---------------------------------------------------------------------------
def _rmse(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.inf
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def run_grid_search(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Exhaustive grid search over N_REGIMES × N_QUANTILES × MIN_MUNI.

    Parameters
    ----------
    df : pd.DataFrame
        Full panel with columns CVEGEO, year, crime_rate.
        Must contain both YEAR_BASE and YEAR_TARGET rows.
    verbose : bool
        Print each trial result.

    Returns
    -------
    pd.DataFrame
        All trials sorted by RMSE ascending.
    """
    observed = (
        df[df['year'] == YEAR_TARGET][['CVEGEO', 'crime_rate']]
        .drop_duplicates('CVEGEO')
        .rename(columns={'crime_rate': 'observed'})
    )

    if observed.empty:
        raise ValueError(f"No rows found for YEAR_TARGET={YEAR_TARGET}. "
                         "Check that your df contains the target year.")

    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"Grid search: {total} combinations "
          f"({' × '.join(str(len(v)) for v in values)})\n")

    results = []

    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        label  = (f"N_REGIMES={params['N_REGIMES']}  "
                  f"N_QUANTILES={params['N_QUANTILES']}  "
                  f"MIN_MUNI={params['MIN_MUNI']}")

        try:
            forecast = _run_forecast(
                df_input    = df,
                N_REGIMES   = params['N_REGIMES'],
                N_QUANTILES = params['N_QUANTILES'],
                MIN_MUNI    = params['MIN_MUNI'],
            )

            merged = observed.merge(forecast, on='CVEGEO', how='inner')
            if merged.empty:
                raise ValueError("No CVEGEO overlap between forecast and observed.")

            rmse = _rmse(
                merged['observed'].values,
                merged['markov_fuzzy'].values
            )

            results.append({**params, 'n_municipalities': len(merged), 'RMSE': round(rmse, 4)})

            if verbose:
                print(f"[{idx:>3}/{total}]  {label}  →  RMSE={rmse:.4f}  "
                      f"(n={len(merged)})")

        except Exception as e:
            results.append({**params, 'n_municipalities': np.nan, 'RMSE': np.inf})
            if verbose:
                print(f"[{idx:>3}/{total}]  {label}  →  FAILED: {e}")

    results_df = (
        pd.DataFrame(results)
        .sort_values('RMSE')
        .reset_index(drop=True)
    )

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from src.data_loader import *   # adjust to your actual loader

    df = crime_rates()                        # must return the full panel df

    results = run_grid_search(df, verbose=True)

    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)
    print(results.head(10).to_string(index=True))

    best = results.iloc[0]
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"  N_REGIMES   = {int(best['N_REGIMES'])}")
    print(f"  N_QUANTILES = {int(best['N_QUANTILES'])}")
    print(f"  MIN_MUNI    = {int(best['MIN_MUNI'])}")
    print(f"  RMSE        = {best['RMSE']}")
    print(f"  n_muni      = {int(best['n_municipalities'])}")

    results.to_csv("data/interim/fuzzy_markov_grid_search.csv", index=False)
    print("\nFull results saved to data/interim/fuzzy_markov_grid_search.csv")
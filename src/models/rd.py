import pandas as pd
import numpy as np
from scipy.optimize import minimize


def rd(df_function, MATRIX, START_YEAR=2015, YEAR_BASE=2024):
    """
    Reaction-diffusion forecast for all municipalities.
    Iterates from START_YEAR to YEAR_BASE (inclusive) using:
        H[t+1] = H[t] + r * H[t] * (1 - H[t] / k) + a * (neigh_avg[t] - H[t])
    Calibrates r, k, a by minimizing RMSE over the full historical series.
    Saves data/interim/rd.csv with columns [CVEGEO, rd].
    """

    base_rates = df_function                      
    base_rates['CVEGEO'] = base_rates['CVEGEO'].astype(str).str.zfill(5)

    # Pivot to wide: rows = municipalities, columns = years
    wide = base_rates.pivot(index='CVEGEO', columns='year', values='crime_rate')
    wide = wide.sort_index()

    years = [y for y in range(START_YEAR, YEAR_BASE + 1) if y in wide.columns]

    W = MATRIX
    W.index = W.index.astype(str).str.zfill(5)
    W.columns = W.columns.astype(str).str.zfill(5)

    # Filter munis to those present in W
    munis = wide.index.tolist()
    munis = [m for m in munis if m in W.index]

    # Re-slice wide AFTER filtering
    wide = wide.loc[munis]
    obs = wide[years].values.astype(float)
    N = len(munis)

    H0 = np.nan_to_num(obs[:, 0], nan=0.0)

    # Align W to filtered munis
    W = W.loc[munis, munis].values.astype(float)

    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_norm = W / row_sums

    def simulate(r, k, a, H0):
        """Run forward simulation starting from H0 for len(years) steps."""
        T = len(years)
        H = np.zeros((N, T))
        H[:, 0] = H0
        for t in range(T - 1):
            neigh_avg = W_norm @ H[:, t]
            reaction  = r * H[:, t] * (1.0 - H[:, t] / k)
            diffusion = a * (neigh_avg - H[:, t])
            H[:, t + 1] = np.clip(H[:, t] + reaction + diffusion, 0, None)
        return H

    # Initial conditions: observed rate at START_YEAR (fill NaN with 0)
    H0 = np.nan_to_num(obs[:, 0], nan=0.0)

    # Mask for observed values (skip NaN cells in RMSE)
    obs_mask = ~np.isnan(obs)

    def objective(params):
        r, k, a = params
        H_sim = simulate(r, k, a, H0)
        residuals = (H_sim - obs)[obs_mask]
        return np.sqrt(np.mean(residuals ** 2))

    initial_guess = [
        0.1,
        wide.max().max() * 1.1,
        0.05,
    ]

    bounds = [
        (0.0, 1.0),                   # r
        (1.0, wide.max().max() * 5),  # k
        (0.0, 1.0),                   # a
    ]

    print("Calibrating reaction-diffusion model...")
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    r_opt, k_opt, a_opt = result.x

    print(f"  Calibration RMSE : {result.fun:.4f}")
    print(f"  r = {r_opt:.4f}  |  k = {k_opt:.4f}  |  a = {a_opt:.4f}")

    H_final = simulate(r_opt, k_opt, a_opt, H0)       # shape (N, T)
    forecast_2025 = H_final[:, -1]                    # last column = YEAR_BASE

    out = pd.DataFrame({'CVEGEO': munis, 'rd': np.round(forecast_2025, 2)})
    out.to_csv('data/interim/rd.csv', index=False)
    print(f"Saved {len(out)} rows to data/interim/rd.csv")

    return out


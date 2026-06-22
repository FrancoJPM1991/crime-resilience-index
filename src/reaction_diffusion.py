import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from reaction_classic import compute_reaction
import numpy as np
from diffusion_classic import compute_diffusion, row_normalise

def compute_reaction(H: pd.Series, r: float, k: float) -> pd.Series:
    return r * H * (1.0 - H / k)

def row_normalise(W: pd.DataFrame) -> pd.DataFrame:
    row_sums = W.sum(axis=1)
    row_sums_safe = row_sums.replace(0, np.nan)
    return W.div(row_sums_safe, axis=0).fillna(0.0)

def compute_diffusion(H: pd.Series, W_norm: pd.DataFrame, a: float) -> pd.Series:
    H_values = H.values  
    neigh_avg = W_norm.values @ H_values          # shape (n,)
    neigh_avg = pd.Series(neigh_avg, index=W_norm.index)
    D = a * (neigh_avg - H)
    return D

def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["CVEGEO"] = df["CVEGEO"].astype(str).str.zfill(5)
    return df[["CVEGEO", "year", "tasa_violencia"]].copy()


def load_weight_matrix(path: Path, cvegeos: list[str]) -> pd.DataFrame:
    W_raw = pd.read_csv(path, index_col=0)
    W_raw.index   = W_raw.index.astype(str).str.zfill(5)
    W_raw.columns = W_raw.columns.astype(str).str.zfill(5)

    # Reindex to exactly the municipalities present in the panel
    W = W_raw.reindex(index=cvegeos, columns=cvegeos).fillna(0.0)
    return row_normalise(W)


def pivot_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot(index="CVEGEO", columns="year", values="tasa_violencia")


# ── Core simulation ────────────────────────────────────────────────────────

def simulate(H_init: pd.Series, W_norm: pd.DataFrame,
             r: float, k: float, a: float,
             n_steps: int) -> list[pd.Series]:
    """
    Run the model forward n_steps steps from H_init.

    Returns
    -------
    list of pd.Series, length n_steps + 1  (includes the initial state)
    """
    states = [H_init.copy()]
    H = H_init.copy()
    for _ in range(n_steps):
        R = compute_reaction(H, r, k)
        D = compute_diffusion(H, W_norm, a)
        H = H + R + D
        H = H.clip(lower=0.0)          # rates cannot go negative
        states.append(H.copy())
    return states

# -- Calibrate -------------------------------------------------------------
def objective(params, wide, W_norm):

    r, k, a = params

    # avoid pathological values
    if r < 0 or k <= 0 or a < 0:
        return 1e20

    sse = 0

    for year in range(START_Y, END_Y):

        H = wide[year].fillna(0)

        R = compute_reaction(H, r, k)
        D = compute_diffusion(H, W_norm, a)

        H_pred = (H + R + D).clip(lower=0)

        H_true = wide[year + 1].fillna(0)

        sse += np.sum((H_true - H_pred) ** 2)

    return sse

def calibrate_model(wide, W_norm):

    initial_guess = [
        0.1,
        wide.max().max() * 1.1,
        0.05
    ]

    bounds = [
        (0.0, 1.0),                    # r
        (1.0, wide.max().max() * 5),  # K
        (0.0, 1.0)                    # a
    ]

    result = minimize(
        objective,
        initial_guess,
        args=(wide, W_norm),
        method="L-BFGS-B",
        bounds=bounds
    )

    return result.x

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    panel  = load_panel(PANEL_PATH)
    wide   = pivot_panel(panel)
    cvegeos = wide.index.tolist()
    W_norm  = load_weight_matrix(ADJ_PATH, cvegeos)

    print("Calibrating parameters...")

    R_fit, K_fit, A_fit = calibrate_model(wide, W_norm)

    print(f"Estimated r = {R_fit:.4f}")
    print(f"Estimated K = {K_fit:.4f}")
    print(f"Estimated a = {A_fit:.4f}")

    # ── In-sample run: step through each observed transition ──────────────
    records = []
    H_obs_start = wide[START_Y].fillna(0.0)

    H = H_obs_start.copy()
    for year in range(START_Y, END_Y + 1):
        records.append(
            pd.Series(H.values, index=cvegeos, name=year)
        )
        if year < END_Y:
            Ri = compute_reaction(H, R_fit, K_fit)
            Di = compute_diffusion(H, W_norm, A_fit)
            H = (H + Ri + Di).clip(lower=0.0)

    # ── Out-of-sample: predict 2025 from last observed state ─────────────
    H_2024_obs = wide[END_Y].fillna(0.0)
    states_2025 = simulate(H_2024_obs, W_norm, R_fit, K_fit, A_fit, n_steps=1)
    H_pred_2025 = states_2025[1]
    H_pred_2025.name = PRED_Y
    records.append(H_pred_2025)

    # ── Assemble output ────────────────────────────────────────────────────
    predictions = pd.concat(records, axis=1)
    predictions.index.name = "CVEGEO"
    predictions.to_csv(OUTPUT_PATH)
    print(f"Saved predictions to {OUTPUT_PATH}")
    print(predictions[[END_Y, PRED_Y]].describe())

    return predictions, wide
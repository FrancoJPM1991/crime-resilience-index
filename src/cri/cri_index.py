import numpy as np
import pandas as pd
from src.config import CRI_ALPHA, CRI_BETA, CRI_GAMMA, SPATIAL_LAMBDA, USE_SPATIAL_LAG


def _row_standardize(W: pd.DataFrame) -> pd.DataFrame:
    """
    Row-standardize a spatial weight matrix so each row sums to 1.
    Rows with zero sum (isolated municipalities) are left as zero.
    """
    row_sums = W.sum(axis=1)
    W_std = W.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    return W_std


def _apply_spatial_lag(
    cri_series: pd.Series,
    W: pd.DataFrame,
    spatial_lambda: float
) -> pd.Series:
    """
    Apply spatial lag to CRI scores:
        CRI* = (1 - λ) * CRI + λ * W * CRI

    Parameters
    ----------
    cri_series : pd.Series
        Raw normalized CRI scores indexed by CVEGEO.
    W : pd.DataFrame
        Row-standardized spatial weight matrix with CVEGEO as index and columns.
    spatial_lambda : float
        Spatial smoothing parameter in [0, 1].

    Returns
    -------
    pd.Series of spatially smoothed CRI scores indexed by CVEGEO.
    """
    # Align CRI series with matrix index — municipalities not in W get zero lag
    cri_aligned = cri_series.reindex(W.index, fill_value=0.0)

    # Spatial lag: weighted average of neighbors' CRI scores
    spatial_component = W.values @ cri_aligned.values
    spatial_lag = pd.Series(spatial_component, index=W.index)

    # Blend own score with spatial lag
    cri_smoothed = (
        (1 - spatial_lambda) * cri_series
        + spatial_lambda * spatial_lag.reindex(cri_series.index, fill_value=0.0)
    )

    return cri_smoothed


def assemble_cri(
    components_df: pd.DataFrame,
    spatial_matrix: pd.DataFrame | None = None,
    alpha: float = CRI_ALPHA,
    beta: float = CRI_BETA,
    gamma: float = CRI_GAMMA,
    spatial_lambda: float = SPATIAL_LAMBDA
) -> pd.DataFrame:
    """
    Assemble the Crime Resilience Index from raw components.

    Steps:
        1. Weighted composite: CRI_raw = α·TRS + β·SCS + γ·PP
        2. Min-max normalization to [0, 1]
        3. Optional spatial lag using provided weight matrix

    Parameters
    ----------
    components_df : pd.DataFrame
        Output of compute_cri_components() with columns TRS, SCS, PP, CVEGEO.
    spatial_matrix : pd.DataFrame or None
        Spatial weight matrix (w_contiguity or w_density) with CVEGEO as
        index and columns. If None, spatial lag is skipped regardless of
        USE_SPATIAL_LAG config flag.
    alpha : float
        Weight for TRS (default CRI_ALPHA = 0.5).
    beta : float
        Weight for SCS (default CRI_BETA = 0.3).
    gamma : float
        Weight for PP  (default CRI_GAMMA = 0.2).
    spatial_lambda : float
        Spatial smoothing parameter λ in [0, 1] (default SPATIAL_LAMBDA = 0.2).

    Returns
    -------
    pd.DataFrame with all component columns plus:
        CRI_raw       : weighted composite before normalization
        CRI           : min-max normalized to [0, 1]
        CRI_spatial   : spatially smoothed CRI (if spatial_matrix provided)
    """
    print("\n*** Assembling CRI ***")
    print(f"  Weights — α(TRS)={alpha}, β(SCS)={beta}, γ(PP)={gamma}")
    assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
        f"Weights must sum to 1.0, got {alpha + beta + gamma:.4f}"

    df = components_df.copy()

    # --- Step 1: Weighted composite ---
    df['CRI_raw'] = (
        alpha * df['TRS']
        + beta  * df['SCS']
        + gamma * df['PP']
    )

    # --- Step 2: Min-max normalization ---
    cri_min = df['CRI_raw'].min()
    cri_max = df['CRI_raw'].max()
    df['CRI'] = (df['CRI_raw'] - cri_min) / (cri_max - cri_min)

    print(f"  CRI_raw range: [{cri_min:.4f}, {cri_max:.4f}]")
    print(f"  CRI normalized range: [0.0000, 1.0000]")

    # --- Step 3: Optional spatial lag ---
    if spatial_matrix is not None and USE_SPATIAL_LAG:
        matrix_name = "provided matrix"
        print(f"  Applying spatial lag (λ={spatial_lambda}) using {matrix_name}...")

        W = _row_standardize(spatial_matrix.copy())

        # Ensure CVEGEO index alignment
        W.index   = W.index.astype(str).str.zfill(5)
        W.columns = W.columns.astype(str).str.zfill(5)

        cri_series = df.set_index('CVEGEO')['CRI']
        cri_smoothed = _apply_spatial_lag(cri_series, W, spatial_lambda)

        # Re-normalize after spatial smoothing to keep [0, 1] bounds
        cri_smoothed = (cri_smoothed - cri_smoothed.min()) / \
                       (cri_smoothed.max() - cri_smoothed.min())

        df['CRI_spatial'] = df['CVEGEO'].map(cri_smoothed)
        print(f"  CRI_spatial range: [{df['CRI_spatial'].min():.4f}, "
              f"{df['CRI_spatial'].max():.4f}]")
    else:
        df['CRI_spatial'] = np.nan
        if spatial_matrix is None:
            print("  Spatial lag skipped — no matrix provided.")
        else:
            print("  Spatial lag skipped — USE_SPATIAL_LAG=False in config.")

    # --- Diagnostic summary ---
    print("\n  CRI summary by regime:")
    summary = (
        df.groupby('regime')[['CRI', 'CRI_spatial']]
        .agg(['mean', 'std'])
        .round(4)
    )
    print(summary)

    print(f"\n  Total municipalities in index: {len(df)}")
    print("*** CRI assembled successfully ***\n")

    return df
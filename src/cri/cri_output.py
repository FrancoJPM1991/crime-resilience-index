import pandas as pd
import numpy as np
from src.config import OUTPUT_DIR


REGIME_LABELS = {
    'df_regime1': 'Very Low Violence',
    'df_regime2': 'Low Violence',
    'df_regime3': 'Moderate Violence',
    'df_regime4': 'High Violence',
    'df_regime5': 'Very High Violence',
}

CRI_LABELS = {
    (0.75, 1.00): 'High Resilience',
    (0.50, 0.75): 'Moderate Resilience',
    (0.25, 0.50): 'Moderate Vulnerability',
    (0.00, 0.25): 'High Vulnerability',
}


def _assign_cri_label(score: float) -> str:
    """Assign interpretive label based on CRI score."""
    for (lo, hi), label in CRI_LABELS.items():
        if lo <= score <= hi:
            return label
    return 'Unclassified'


def export_cri(cri_df: pd.DataFrame, use_spatial: bool = True) -> None:
    """
    Export CRI results to CSV and print a structured summary report.

    Parameters
    ----------
    cri_df : pd.DataFrame
        Output of assemble_cri() containing CVEGEO, regime, current_state,
        crime_rate_2024, TRS, SCS, PP, CRI_raw, CRI, and CRI_spatial columns.
    use_spatial : bool
        If True and CRI_spatial is available, use CRI_spatial as the
        primary score for labeling and summary. Default True.
    """
    print("\n*** Exporting CRI results ***")

    df = cri_df.copy()

    # --- Determine primary CRI column ---
    has_spatial = use_spatial and df['CRI_spatial'].notna().all()
    primary_col = 'CRI_spatial' if has_spatial else 'CRI'
    print(f"  Primary CRI column: {primary_col}")

    # --- Add interpretive columns ---
    df['regime_label'] = df['regime'].map(REGIME_LABELS)
    df['CRI_label']    = df[primary_col].apply(_assign_cri_label)

    # --- Column order for output ---
    export_cols = [
        'CVEGEO', 'regime', 'regime_label',
        'current_state', 'crime_rate_2024',
        'D', 'U', 'pi', 'G',
        'TRS', 'SCS', 'PP',
        'CRI_raw', 'CRI', 'CRI_spatial',
        'CRI_label'
    ]
    export_cols = [c for c in export_cols if c in df.columns]

    # --- Save full results ---
    output_path = f"{OUTPUT_DIR}/cri.csv"
    df[export_cols].to_csv(output_path, index=False)
    print(f"  Full results saved to {output_path}")

    # --- Print summary report ---
    _print_summary(df, primary_col)


def _print_summary(df: pd.DataFrame, primary_col: str) -> None:
    """Print a structured CRI summary report to console."""

    sep = "=" * 65

    print(f"\n{sep}")
    print("  CRIME RESILIENCE INDEX — SUMMARY REPORT")
    print(sep)

    # Overall distribution
    print(f"\n  Municipalities in index : {len(df)}")
    print(f"  CRI mean  : {df[primary_col].mean():.4f}")
    print(f"  CRI std   : {df[primary_col].std():.4f}")
    print(f"  CRI min   : {df[primary_col].min():.4f}")
    print(f"  CRI max   : {df[primary_col].max():.4f}")

    # By resilience label
    print(f"\n  Distribution by resilience category ({primary_col}):")
    label_counts = df['CRI_label'].value_counts()
    label_pct    = (label_counts / len(df) * 100).round(1)
    for label in ['High Resilience', 'Moderate Resilience',
                  'Moderate Vulnerability', 'High Vulnerability']:
        count = label_counts.get(label, 0)
        pct   = label_pct.get(label, 0.0)
        print(f"    {label:<25} {count:>5} municipalities  ({pct:>5.1f}%)")

    # By regime
    print(f"\n  CRI statistics by violence regime:")
    print(f"  {'Regime':<22} {'N':>5}  {'Mean':>7}  {'Std':>7}  "
          f"{'Min':>7}  {'Max':>7}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    regime_order = [f'df_regime{i}' for i in range(1, 6)]
    for regime in regime_order:
        subset = df[df['regime'] == regime]
        if len(subset) == 0:
            continue
        label = REGIME_LABELS.get(regime, regime)
        n     = len(subset)
        mean  = subset[primary_col].mean()
        std   = subset[primary_col].std()
        mn    = subset[primary_col].min()
        mx    = subset[primary_col].max()
        print(f"  {label:<22} {n:>5}  {mean:>7.4f}  {std:>7.4f}  "
              f"{mn:>7.4f}  {mx:>7.4f}")

    # Top and bottom 10
    print(f"\n  Top 10 most resilient municipalities:")
    top10 = (
        df.nlargest(10, primary_col)
        [['CVEGEO', 'regime_label', 'current_state', 'crime_rate_2024', primary_col]]
    )
    print(top10.to_string(index=False))

    print(f"\n  Top 10 most vulnerable municipalities:")
    bot10 = (
        df.nsmallest(10, primary_col)
        [['CVEGEO', 'regime_label', 'current_state', 'crime_rate_2024', primary_col]]
    )
    print(bot10.to_string(index=False))

    print(f"\n{sep}\n")
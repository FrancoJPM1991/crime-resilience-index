import numpy as np
from src.ols import ols
from src.ma import ma

def hybrid(df_function, markov, model_name, YEAR_BASE=2024, STD_THRESHOLD=2, MA_WINDOW=3):
    df = df_function.copy()
    hybrid_df = markov.copy()

    # 1. FIXED: Store function references without parentheses '()'
    model_mapping = {
        'OLS': ols,
        'MA': ma
    }

    selected_function = model_mapping.get(model_name.upper())

    if selected_function is None:
        raise ValueError(f"Invalid model_name '{model_name}'. Choose from {list(model_mapping.keys())}")

    # Calculate Z-Scores
    mun_means = (
        df.groupby(['CVEGEO', 'regimes'])['crime_rate']
        .mean()
        .reset_index()
        .rename(columns={'crime_rate': 'mun_mean'})
    )
    regime_stats = (
        mun_means.groupby('regimes')['mun_mean']
        .agg(regime_mean='mean', regime_std='std')
        .reset_index()
    )
    mun_means = mun_means.merge(regime_stats, on='regimes')
    mun_means['z_score'] = (
        (mun_means['mun_mean'] - mun_means['regime_mean']) / mun_means['regime_std']
    )

    outliers = mun_means[mun_means['z_score'] > STD_THRESHOLD]['CVEGEO'].tolist()
    print(f"\nOutlier anchor — {len(outliers)} municipalities flagged "
          f"(z > {STD_THRESHOLD} within their regime):")
    
    target_col = 'markov'

    # 2. FIXED: Execute the functions dynamically inside the loop
    for mun in outliers:
        mask = hybrid_df['CVEGEO'] == mun
        if mask.any():
            
            # Call the function conditionally depending on what arguments it requires
            if model_name.upper() == 'MA':
                estimate = selected_function(mun, df, YEAR_BASE=YEAR_BASE, MA_WINDOW=MA_WINDOW)
            else:
                estimate = selected_function(mun, df, YEAR_BASE=YEAR_BASE)
                
            hybrid_df.loc[mask, target_col] = round(estimate, 2)

    hybrid_df = hybrid_df.rename(columns={'markov': 'hybrid'})

    return hybrid_df
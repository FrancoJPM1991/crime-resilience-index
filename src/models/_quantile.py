import pandas as pd
import numpy as np

def assign_quantiles(regime_dfs: dict, N_QUANTILES=5):
    
    result = {}
    for name, df in regime_dfs.items():
        df = df.copy()

        # First pass: discover how many unique bin edges survive after deduplication
        _, bins = pd.qcut(
            df['crime_rate'],
            q=N_QUANTILES,
            retbins=True,
            duplicates='drop'
        )
        actual_n = len(bins) - 1  # real number of bins after deduplication

        if actual_n < N_QUANTILES:
            print(f"  {name}: requested {N_QUANTILES} quantiles but only "
                  f"{actual_n} unique bin edges found using {actual_n} states.")

        # Second pass: assign labels now that we know actual_n
        df['state'] = pd.qcut(
            df['crime_rate'],
            q=N_QUANTILES,
            labels=list(range(1, actual_n + 1)),
            duplicates='drop'
        ).astype(int)

        # Mean violence rate per state used later for defuzzification
        state_means = (
            df.groupby('state')['crime_rate']
            .mean()
            .rename('state_mean')
        )
        df = df.join(state_means, on='state')

        actual_states = sorted(df['state'].unique())
        print(f"{name}: {len(actual_states)} states | bins: {np.round(bins, 1)}")
        print(f"  State means: { {s: round(state_means[s], 1) for s in actual_states} }")
        result[name] = df

    return result
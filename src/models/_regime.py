import pandas as pd
import numpy as np

def regime_construction(df_function, N_REGIMES=5):
    df = df_function.copy()
    df['log_crime_rate'] = np.log1p(df['crime_rate'])

    min_log = df['log_crime_rate'].min()
    max_log = df['log_crime_rate'].max()

    df['norm_crime_rate'] = (df['log_crime_rate'] - min_log) / (max_log - min_log)

    df['regimes'] = pd.cut(df['norm_crime_rate'], bins=N_REGIMES, labels=list(range(N_REGIMES + 1)[1:]))

    df.to_csv("data/interim/crime_rates_regimes.csv", index=False)

    regime_dfs = {}

    for regime in sorted(df['regimes'].unique().tolist()):
        df_name = f"df_regime{regime}"
        regime_dfs[df_name] = df[df['regimes'] == regime].copy()

    for name, df in regime_dfs.items():
        df['crime_rate'] = df['crime_rate'].astype(int)
        max_val = df['crime_rate'].max()
        min_val = df['crime_rate'].min()

        print(f"The violence regime {name} max: {max_val} y min: {min_val}")
        print(f"The length of {name} is {len(df)}")

    return regime_dfs


import pandas as pd
import numpy as np

def transition_matrix(regime_dfs: dict):
    results = {}
    for name, df in regime_dfs.items():
        df = df.sort_values(['CVEGEO', 'year'])

        # Build (state_t, state_t+1) pairs within each municipality
        df['state_next'] = df.groupby('CVEGEO')['state'].shift(-1)
        transitions = df.dropna(subset=['state_next']).copy()
        transitions['state_next'] = transitions['state_next'].astype(int)

        states = sorted(df['state'].unique())
        n = len(states)
        state_index = {s: i for i, s in enumerate(states)}

        # Count matrix
        counts = np.zeros((n, n), dtype=float)
        for _, row in transitions.iterrows():
            i = state_index[row['state']]
            j = state_index[int(row['state_next'])]
            counts[i, j] += 1

        # Row-normalize; rows with no observations stay uniform
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T = counts / row_sums

        T_df = pd.DataFrame(T, index=states, columns=states)
        T_df.index.name   = 'from_state'
        T_df.columns.name = 'to_state'
        T_df.to_csv(f"data/interim/T_{name}.csv")

        # State means (one value per state, for defuzzification later)
        state_means = (
            df.groupby('state')['crime_rate'].mean()
        )

        print(f"\n{name} — transition matrix ({n}×{n}):")
        print(T_df.round(3))
        results[name] = {'T': T_df, 'state_means': state_means, 'states': states}

    return results
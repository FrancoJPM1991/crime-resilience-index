import pandas as pd
import numpy as np


def transition_matrix(regime_dfs: dict):
    results = {}
    for name, df in regime_dfs.items():
        df = df.sort_values(['CVEGEO', 'year'])

        # Build (state_t, state_t+1) pairs within each municipality
        df['state_next'] = df.groupby('CVEGEO')['state'].shift(-1)
        df['membership_next'] = df.groupby('CVEGEO')['membership'].shift(-1)
        transitions = df.dropna(subset=['state_next']).copy()
        transitions['state_next'] = transitions['state_next'].astype(int)

        states = sorted(df['state'].unique())
        n = len(states)
        state_index = {s: i for i, s in enumerate(states)}

        # Fuzzy-weighted count matrix:
        # each transition (t -> t+1) contributes μ_i(t) to row i,
        # distributed across columns j by μ_j(t+1)
        counts = np.zeros((n, n), dtype=float)
        for _, row in transitions.iterrows():
            mu_from = row['membership']   # dict {state: membership} at time t
            mu_to   = row['membership_next']  # dict {state: membership} at t+1

            if not isinstance(mu_from, dict) or not isinstance(mu_to, dict):
                continue

            for s_from, w_from in mu_from.items():
                if s_from not in state_index:
                    continue
                i = state_index[s_from]
                for s_to, w_to in mu_to.items():
                    if s_to not in state_index:
                        continue
                    j = state_index[s_to]
                    counts[i, j] += w_from * w_to

        # Row-normalise; rows with no observations stay uniform
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T = counts / row_sums

        T_df = pd.DataFrame(T, index=states, columns=states)
        T_df.index.name   = 'from_state'
        T_df.columns.name = 'to_state'

        # State means (one value per state, for defuzzification later)
        state_means = df.groupby('state')['crime_rate'].mean()

        print(f"\n{name} — fuzzy transition matrix ({n}×{n}):")
        print(T_df.round(3))
        results[name] = {'T': T_df, 'state_means': state_means, 'states': states}

    return results
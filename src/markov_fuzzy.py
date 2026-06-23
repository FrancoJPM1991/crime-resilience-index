import pandas as pd
import numpy as np
from src.regime import regime_construction
from src.quantile_fuzzy import assign_quantiles
from src.transition_fuzzy import transition_matrix
from src.thin_regimes import merge_thin_regimes


def markov_prediction_fuzzy(df_function, YEAR_BASE=2024, N_STEPS=1, N_REGIMES=5, N_QUANTILES=5, MIN_MUNI=30, BINS=5):

    regime_dfs = regime_construction(df_function, N_REGIMES)
    merged_regimes_dfs = merge_thin_regimes(regime_dfs, MIN_MUNI)
    quantile_dfs = assign_quantiles(merged_regimes_dfs, N_QUANTILES, BINS)
    transition_results = transition_matrix(quantile_dfs)

    records = []

    for reg_name, df in quantile_dfs.items():
        T_data = transition_results[reg_name]
        T = T_data['T'].values
        states = T_data['states']
        state_means = T_data['state_means']

        base = df[df['year'] == YEAR_BASE][['CVEGEO', 'state', 'membership']].drop_duplicates('CVEGEO')

        for _, row in base.iterrows():
            mun = row['CVEGEO']
            membership = row['membership']  # dict {state: membership}

            # Initial distribution: fuzzy membership vector (replaces one-hot)
            pi = np.zeros(len(states))
            if isinstance(membership, dict):
                for s, mu in membership.items():
                    if s in states:
                        pi[states.index(s)] = mu
            else:
                # Fallback to one-hot if membership missing
                s0 = int(row['state'])
                pi[states.index(s0)] = 1.0

            # Normalise pi (should already sum to 1, but guard against float drift)
            pi_sum = pi.sum()
            if pi_sum > 0:
                pi = pi / pi_sum

            # Propagate n steps
            T_n  = np.linalg.matrix_power(T, N_STEPS)
            pi_n = pi @ T_n  # shape (n_states,)

            # Crisp forecast: weighted mean of state means
            crisp = float(np.dot(pi_n, [state_means[s] for s in states]))

            # Most likely future state
            forecast_state = states[int(np.argmax(pi_n))]

            records.append({
                'CVEGEO':         mun,
                'regime':          reg_name,
                'forecast_state':  forecast_state,
                'forecast_rate':   round(crisp, 2),
                'pi_n':            dict(zip(states, pi_n.round(4)))  # full distribution
            })

    markov = pd.DataFrame(records)
    markov = markov.rename(columns={'forecast_rate': 'markov_fuzzy'})
    print(f"\nMarkov forecast summary (n_steps={N_STEPS}, base_year={YEAR_BASE}):")
    print(markov.groupby('regime')['markov_fuzzy'].describe().round(1))
    return markov
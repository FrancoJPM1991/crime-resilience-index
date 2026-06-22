import pandas as pd
import numpy as np
from src.regime import regime_construction
from src.quantile import assign_quantiles
from src.transition import transition_matrix

def markov_prediction(df_function, YEAR_BASE=2024, N_STEPS=1, N_REGIMES=5, N_QUANTILES=5):

    regime_dfs = regime_construction(df_function, N_REGIMES)
    quantile_dfs = assign_quantiles(regime_dfs, N_QUANTILES)
    transition_results = transition_matrix(quantile_dfs)

    records = []

    for reg_name, df in quantile_dfs.items():
        T_data = transition_results[reg_name]
        T = T_data['T'].values
        states = T_data['states']
        state_means = T_data['state_means']

        base =df[df['year'] == YEAR_BASE][['CVEGEO', 'state']].drop_duplicates('CVEGEO')

        for _, row in base.iterrows():
            mun = row['CVEGEO']
            s0 = int(row['state'])

            # Initial distribution: deterministic (one-hot)
            idx   = states.index(s0)
            pi    = np.zeros(len(states))
            pi[idx] = 1.0

            # Propagate n steps
            T_n   = np.linalg.matrix_power(T, N_STEPS)
            pi_n  = pi @ T_n                     # shape (n_states,)

            # Crisp forecast: weighted mean of state means
            crisp = float(np.dot(pi_n, [state_means[s] for s in states]))

            # Most likely future state
            forecast_state = states[int(np.argmax(pi_n))]

            records.append({
                'CVEGEO':        mun,
                'regime':         reg_name,
                'forecast_state': forecast_state,
                'forecast_rate':  round(crisp, 2)
            })
    
    markov = pd.DataFrame(records)
    print(f"\nMarkov forecast summary (n_steps={N_STEPS}, base_year={YEAR_BASE}):")
    print(markov.groupby('regime')['forecast_rate'].describe().round(1))
    return markov


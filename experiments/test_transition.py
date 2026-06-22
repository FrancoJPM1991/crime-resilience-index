import pandas as pd
from src.transition import transition_matrix
from src.regime import regime_construction
from src.quantile import assign_quantiles
from src.data_loader import crime_rates

df = crime_rates()
regime_dfs = regime_construction(df, N_REGIMES=5)
quantile_dfs = assign_quantiles(regime_dfs, N_QUANTILES=5)
transition_dfs = transition_matrix(quantile_dfs)


print("Transition matrix test completed successfully.")
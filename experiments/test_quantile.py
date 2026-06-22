import pandas as pd
from src.quantile import assign_quantiles
from src.regime import regime_construction
from src.data_loader import crime_rates, crime_rates_regimes


df = crime_rates()
regime_dfs = regime_construction(df, N_REGIMES=5)
quantile_dfs = assign_quantiles(regime_dfs, N_QUANTILES=5)

print("Quantile assignment test completed successfully.")

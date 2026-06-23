import pandas as pd
import numpy as np
from src.ols import ols
from src.ma import ma
from src.rd import rd
from src.markov import markov_prediction
from src.data_loader import *
from src.hybrid import hybrid
from src.config import *
from src.validation import validate

base_rates = crime_rates()
regime_rates = crime_rates_regimes()

markov = markov_prediction(crime_rates(), YEAR_BASE, N_STEPS, N_REGIMES, N_QUANTILES)
predictions_df = hybrid(crime_rates_regimes(), markov, AUX_MODEL, YEAR_BASE, STD_THRESHOLD, MA_WINDOW)
predictions_df = predictions_df.merge(markov, on='CVEGEO', how='left')

#----- Vectorized benchmarks ---
print("Calculating comparative benchmarks for all municipalities...")
ols_series = (
    base_rates.groupby('CVEGEO')
    .apply(lambda g: ols(g['CVEGEO'].iloc[0], base_rates, YEAR_BASE))
    .rename('ols')
)
ma_series = (
    base_rates.groupby('CVEGEO')
    .apply(lambda g: ma(g['CVEGEO'].iloc[0], base_rates, YEAR_BASE, MA_WINDOW))
    .rename('ma')
)

rd_df = rd(base_rates, MATRIX, START_YEAR, YEAR_BASE)

observed_df = (
    base_rates[base_rates['year'] == OBSERVED_YEAR][['CVEGEO', 'crime_rate']]
    .rename(columns={'crime_rate': 'observed'})
    .assign(observed=lambda x: x['observed'].round(2))
)

persistence_df = (
    base_rates[base_rates['year'] == YEAR_BASE][['CVEGEO', 'crime_rate']]
    .rename(columns={'crime_rate': 'persistence'})
    .assign(persistence=lambda x: x['persistence'].round(2))
)

benchmark_df = (
    predictions_df[['CVEGEO', 'markov', 'hybrid']]
    .merge(ols_series,     on='CVEGEO', how='left')
    .merge(ma_series,      on='CVEGEO', how='left')
    .merge(rd_df,          on='CVEGEO', how='left')
    .merge(persistence_df, on='CVEGEO', how='left')
    .merge(observed_df,    on='CVEGEO', how='left')
)

results = validate(benchmark_df)

benchmark_df.to_csv("results/predictions_benchmark.csv", index=False)
print("\nFinal Predictions & Benchmarks Preview:")
print(predictions_df.head())
print(benchmark_df.head())
print(results)
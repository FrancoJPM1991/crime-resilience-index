import pandas as pd
import numpy as np
from src.config import *
from src.data_loader import crime_rates, crime_rates_regimes
from src.weights import w_contiguity
from src.models.markov import markov_prediction
from src.models.markov_fuzzy import markov_prediction_fuzzy
from src.models.hybrid import hybrid
from src.models.arima import run_arima
from src.models.ols import ols
from src.models.ma import ma
from src.models.rd import rd
from src.validation import validate
from src.cri.cri_weights import build_regime_metadata
from src.cri.cri_components import compute_cri_components
from src.cri.cri_index import assemble_cri
from src.cri.cri_output import export_cri

base_rates = crime_rates()
regime_rates = crime_rates_regimes()
MATRIX = w_contiguity()

print("***************************Calculating comparative benchmarks for all municipalities...********************************")

print("***************************Calculating Markov Chain model***************************")
markov = markov_prediction(crime_rates(), YEAR_BASE, N_STEPS, N_REGIMES, N_QUANTILES)

print("***************************Calculating Fuzzy Markov model***************************")
markov_fuzzy = markov_prediction_fuzzy(crime_rates(), YEAR_BASE, N_STEPS, N_REGIMES, N_QUANTILES, MIN_MUNI, BINS)

print("***************************Calculating Hybrid model***************************")
predictions_df = hybrid(crime_rates_regimes(), markov, AUX_MODEL, YEAR_BASE, STD_THRESHOLD, MA_WINDOW)
predictions_df = (predictions_df.merge(markov, on='CVEGEO', how='left').merge(markov_fuzzy, on='CVEGEO', how='left'))

print("***************************Calculating ordinary least squares***************************")
ols_series = (
    base_rates.groupby('CVEGEO')
    .apply(lambda g: ols(g['CVEGEO'].iloc[0], base_rates, YEAR_BASE))
    .rename('ols')
)

print("***************************Calculating weighted moving average***************************")
wma_series = (
    base_rates.groupby('CVEGEO')
    .apply(lambda g: ma(g['CVEGEO'].iloc[0], base_rates, YEAR_BASE, MA_WINDOW))
    .rename('wma')
)

print("***************************Calculating reaction - diffusion model***************************")
rd_df = rd(base_rates, MATRIX, START_YEAR, YEAR_BASE)


print("***************************Calculating ARIMA model***************************")
arima_series = run_arima(crime_rates(), START_YEAR, YEAR_PREDICT)

print("***************************Adding observed crime rates***************************")
observed_df = (
    base_rates[base_rates['year'] == OBSERVED_YEAR][['CVEGEO', 'crime_rate']]
    .rename(columns={'crime_rate': 'observed'})
    .assign(observed=lambda x: x['observed'].round(2))
)

print("***************************Adding persistence crime rates***************************")
persistence_df = (
    base_rates[base_rates['year'] == YEAR_BASE][['CVEGEO', 'crime_rate']]
    .rename(columns={'crime_rate': 'persistence'})
    .assign(persistence=lambda x: x['persistence'].round(2))
)

benchmark_df = (
    predictions_df[['CVEGEO', 'markov', 'hybrid', 'markov_fuzzy']]
    .merge(ols_series,     on='CVEGEO', how='left')
    .merge(wma_series,     on='CVEGEO', how='left')
    .merge(rd_df,          on='CVEGEO', how='left')
    .merge(arima_series,   on='CVEGEO', how='left')
    .merge(persistence_df, on='CVEGEO', how='left')
    .merge(observed_df,    on='CVEGEO', how='left')
)

results = validate(benchmark_df)


benchmark_df.to_csv("results/predictions_benchmark.csv", index=False)
print("\nFinal Predictions & Benchmarks Preview:")
print(predictions_df.head())
print(benchmark_df.head())
print(results)


print("***************************TESTING CRI***************************")
metadata = build_regime_metadata()
components = compute_cri_components(metadata, YEAR_BASE)
assembly = assemble_cri(components, MATRIX, CRI_ALPHA, CRI_BETA, CRI_GAMMA, SPATIAL_LAMBDA)
export_cri(assembly, True)

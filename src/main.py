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

predictions_df['ols_benchmark'] = np.nan
predictions_df['ma_benchmark'] = np.nan

# 5. FIXED: Correctly loop using row masks to calculate benchmarks per municipality
print("Calculating comparative benchmarks for all municipalities...")
for mun in predictions_df['CVEGEO'].unique():
    mask = predictions_df['CVEGEO'] == mun
    
    # Calculate estimates safely
    ols_val = ols(mun, base_rates, YEAR_BASE)
    ma_val = ma(mun, base_rates, YEAR_BASE, MA_WINDOW)
    
    # Target only the specific municipality's row
    predictions_df.loc[mask, 'ols_benchmark'] = round(ols_val, 2)
    predictions_df.loc[mask, 'ma_benchmark'] = round(ma_val, 2)

# Now your head() will display cleanly without column collision
rd_df = rd(base_rates, MATRIX, START_YEAR, YEAR_BASE)

predictions_df = predictions_df.merge(rd_df, on='CVEGEO', how='left')

observed_df = base_rates[base_rates['year'] == OBSERVED_YEAR].copy()
observed_df = observed_df[['CVEGEO', 'crime_rate']]
observed_df = observed_df.rename(columns={'crime_rate': 'observed'})
observed_df['observed'] = observed_df['observed'].round(2)

persistance_df = base_rates[base_rates['year'] == YEAR_BASE].copy()
persistance_df = persistance_df[['CVEGEO', 'crime_rate']]
persistance_df = persistance_df.rename(columns={'crime_rate': 'persistance'})
persistance_df['persistance'] = persistance_df['persistance'].round(2)

benchmark_df = predictions_df[['CVEGEO', 'markov', 'hybrid', 'ols_benchmark', 'ma_benchmark', 'rd']]
benchmark_df = benchmark_df.merge(persistance_df, on='CVEGEO', how='left')
benchmark_df = benchmark_df.merge(observed_df, on='CVEGEO', how='left')
benchmark_df = benchmark_df.rename(columns={'ols_benchmark': 'ols', 'ma_benchmark': 'ma'})

results = validate(benchmark_df)

benchmark_df.to_csv("results/predictions_benchmark.csv")
print("\nFinal Predictions & Benchmarks Preview:")
print(predictions_df.head())
print(benchmark_df.head())
print(results)
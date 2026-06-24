import pandas as pd
from scipy import stats
import numpy as np

def ma(mun, df_function, YEAR_BASE=2024, MA_WINDOW=3):
    df = df_function.copy()
    history_df = df[
        (df['year'] <= YEAR_BASE) & 
        (df['CVEGEO'] == mun)
    ].sort_values('year')

    history = history_df['crime_rate'].values

    window = min(MA_WINDOW, len(history))
    recent = history[-window:]
    weights = np.arange(1, window + 1, dtype=float)
    ma_est = round(float(np.dot(weights, recent) / weights.sum()), 2)

    return ma_est
 
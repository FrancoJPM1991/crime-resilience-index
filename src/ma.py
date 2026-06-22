import pandas as pd
from scipy import stats
import numpy as np

def ma(mun, df_function, YEAR_BASE=2024, MA_WINDOW=3):
    df = df_function.copy()
    history = df[df['year'] <= YEAR_BASE].copy()
    history = history[history['CVEGEO'] == mun]['crime_rate'].values

    window = min(MA_WINDOW, len(history))
    recent = history[-window:]
    weights = np.arange(1, window + 1)
    ma_est = float(np.dot(recent, weights) / weights.sum())

    return ma_est


                   
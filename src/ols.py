import pandas as pd
from scipy import stats
import numpy as np

def ols(mun, df_function, YEAR_BASE=2024):
    df = df_function.copy()
    history = df[df['year'] <= YEAR_BASE].copy()
    history = history[history['CVEGEO'] == mun]['crime_rate'].values
    x = np.arange(len(history), dtype=float)
    slope, intercept, *_ = stats.linregress(x, history)
    ols_est  = intercept + slope * len(history) 
    ols_est  = max(ols_est, float(history.min()))

    return ols_est
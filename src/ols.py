import pandas as pd
from scipy import stats
import numpy as np

def ols(mun, df_function, YEAR_BASE=2024):
    df = df_function.copy()
    history = df[(df['CVEGEO'] == mun) & (df['year'] <= YEAR_BASE)]['crime_rate'].values

    if len(history) < 2:
        return np.nan

    x = np.arange(len(history), dtype=float)
    result = stats.linregress(x, history)
    ols_est = result.intercept + result.slope * len(history)
    ols_est = round(max(ols_est, 0.0), 2)  

    return ols_est
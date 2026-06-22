import pandas as pd
from scipy import stats
import numpy as np

def ols(df, YEAR_BASE=2024):
    history = df[df['year'] <= YEAR_BASE].copy()
    x = np.arange(len(history), dtype=float)
    slope, intercept, *_ = stats.linregress(x, history)
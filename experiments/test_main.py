import pandas as pd
import numpy as np
from pmdarima import auto_arima
import warnings
from src.data_loader import *

df = crime_rates()

TRAIN_START   = 2015
FORECAST_YEAR = 2025
MIN_OBS       = 5
TARGET_COL    = 'crime_rate'
ID_COL        = 'CVEGEO'
YEAR_COL      = 'year'

train_df = df[(df[YEAR_COL] >= TRAIN_START) & (df[YEAR_COL] < FORECAST_YEAR)].copy()
wide = train_df.pivot_table(index=YEAR_COL, columns=ID_COL, values=TARGET_COL).sort_index()

# Check what's actually coming out of auto_arima on a sample series

cvegeo = list(wide.columns)[5]  # pick one non-constant municipality
series = wide[cvegeo].dropna()



# Check what's actually coming out of auto_arima on a sample series

cvegeo = list(wide.columns)[5]  # pick one non-constant municipality
series = wide[cvegeo].dropna()

print(f"CVEGEO: {cvegeo}")
print(f"Series dtype: {series.dtype}")
print(f"Series index type: {type(series.index)}")
print(f"Values:\n{series.values}")
print()

# Force numpy array — strip the pandas index entirely
import numpy as np
y = series.values.astype(float)

model = auto_arima(
    y,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,
    information_criterion='aic',
    stepwise=True,
    suppress_warnings=False,
    error_action='raise',
)

print(f"Fitted order: {model.order}")
forecast = model.predict(n_periods=1)
print(f"Forecast: {forecast}")
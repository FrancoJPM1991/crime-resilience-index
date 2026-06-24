import pandas as pd
import numpy as np
from pmdarima import auto_arima
import warnings


def _fit_predict_one(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 5:
        return np.nan
    if clean.nunique() == 1:
        return np.nan  # constant series — add this guard too

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                clean.values.astype(float),  # ← strip pandas index
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None,
                information_criterion='aic',
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
            )
        forecast = model.predict(n_periods=1)
        return round(float(forecast[0]), 2)

    except Exception:
        return 0 #np.nan

def run_arima(df: pd.DataFrame, START_YEAR=2015, YEAR_PREDICT=2025) -> pd.Series:
    train_df = df[
        (df['year'] >= START_YEAR) &
        (df['year'] <  YEAR_PREDICT)
    ].copy()

    # Pivot to wide: rows = year, columns = CVEGEO
    wide = (
        train_df
        .pivot_table(index='year', columns='CVEGEO', values='crime_rate')
        .sort_index()
    )

    results = {}
    for cvegeo in wide.columns:
        results[cvegeo] = _fit_predict_one(wide[cvegeo])

    arima_series = (
        pd.Series(results, name='arima')
        .rename_axis('CVEGEO')
        .reset_index()
        .rename(columns={0: 'arima'})
        .set_index('CVEGEO')['arima']
    )

    arima_series = arima_series.fillna(0)
    return arima_series
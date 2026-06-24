import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr


def validate(predictions_df):

    models = ['markov', 'hybrid', 'markov_fuzzy', 'ols', 'wma', 'rd', 'arima', 'persistence']
    obs = predictions_df['observed'].values
    obs_mean  = obs.mean()
    obs_range = obs.max() - obs.min()

    rows = []
    for model in models:
        pred = predictions_df[model].values
        residuals = pred - obs
        abs_res = np.abs(residuals)

        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(abs_res)
        bias = np.mean(residuals)
        cv_rmse = rmse / obs_mean  if obs_mean  != 0 else np.nan
        nrmse = rmse / obs_range if obs_range != 0 else np.nan

        # MAPE: exclude municipalities with zero observed rate
        mask = obs != 0
        mape = np.mean(abs_res[mask] / obs[mask]) * 100 if mask.sum() > 0 else np.nan

        spear, _ = spearmanr(pred, obs)
        pears, _ = pearsonr(pred, obs)

        rows.append({
            'model':        model,
            'RMSE':         round(rmse,    4),
            'MAE':          round(mae,     4),
            'Bias':         round(bias,    4),
            'CV_RMSE':      round(cv_rmse, 4),
            'NRMSE':        round(nrmse,   4),
            'MAPE':         round(mape,    4),
            'Spearman_rho': round(spear,   4),
            'Pearson_r':    round(pears,   4),
        })

    val_df = pd.DataFrame(rows)
    val_df.to_csv('results/validation.csv', index=False)
    print("Saved results/validation.csv")
    print(val_df.to_string(index=False))

    return val_df
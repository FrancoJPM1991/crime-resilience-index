import pandas as pd
import numpy as np
from src.regime import regime_construction
from src.quantile import assign_quantiles
from src.transition import transition_matrix
from src.markov import markov_prediction
from src.data_loader import crime_rates

markov_prediction(crime_rates(), YEAR_BASE=2024, N_STEPS=1, N_REGIMES=5, N_QUANTILES=5)

print("Markov prediction assignment test completed successfully.")


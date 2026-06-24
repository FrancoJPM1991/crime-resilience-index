import pandas as pd
from src.models.ols import ols
from src.data_loader import crime_rates_regimes

mun = "01001"

test_ols = ols(mun, crime_rates_regimes(), YEAR_BASE=2024)

print("The OLS estimate for the crime rate in 2025 for municipality " + mun + " is:", test_ols)
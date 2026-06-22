import pandas as pd
from src.ma import ma
from src.data_loader import crime_rates_regimes

mun = "01001"

test_ma = ma(mun, crime_rates_regimes(), YEAR_BASE=2024, MA_WINDOW=3)

print("The MA estimate for the crime rate in 2025 for municipality " + mun + " is:", test_ma)
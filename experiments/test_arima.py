from src.arima import run_arima, _fit_predict_one
from src.data_loader import *
from src.config import *

arima = run_arima(crime_rates(), START_YEAR, YEAR_PREDICT)

print(arima.head())
print("Test passed succesfully")

from src.regime import regime_construction
from src.data_loader import *

test_regime = regime_construction(crime_rates(), N_REGIMES=5)

print("Test regime construction completed successfully.")
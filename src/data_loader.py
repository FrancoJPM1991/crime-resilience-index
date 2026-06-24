import pandas as pd
from src.config import DATA_INTERIM, DATA_RAW_CRIME

def criminal_activity():
    df = pd.read_csv(f"{DATA_RAW_CRIME}/Municipal-Delitos-2015-2025_abr2026.csv", encoding="latin-1")
    return df

def crime_rates():
    df = pd.read_csv(f"{DATA_INTERIM}/crime_rates_AB.csv", encoding="latin-1")
    df['CVEGEO'] = df['CVEGEO'].astype(str).str.zfill(5)
    return df

def crime_rates_regimes():
    df = pd.read_csv(f"{DATA_INTERIM}/crime_rates_regimes.csv", encoding="latin-1")
    df['CVEGEO'] = df['CVEGEO'].astype(str).str.zfill(5)
    return df
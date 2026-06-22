import pandas as pd

def criminal_activity():
    base_path = f"data/raw/crime"

    file = f"{base_path}/{'Municipal-Delitos-2015-2025_abr2026.csv'}"

    df = pd.read_csv(file, encoding="latin-1")

    return df

def crime_rates():
    base_path = f"data/interim"

    file = f"{base_path}/{'crime_rates.csv'}"

    df = pd.read_csv(file, encoding="latin-1")

    return df


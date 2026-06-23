import pandas as pd

def criminal_activity():
    base_path = f"data/raw/crime"
    file = f"{base_path}/{'Municipal-Delitos-2015-2025_abr2026.csv'}"
    df = pd.read_csv(file, encoding="latin-1")
    return df

def crime_rates():
    base_path = f"data/interim"
    file = f"{base_path}/{'crime_rates_AB.csv'}"
    df = pd.read_csv(file, encoding="latin-1")
    df['CVEGEO'] = df['CVEGEO'].astype(str).str.zfill(5)
    return df

def crime_rates_regimes():
    base_path = f"data/interim"
    file = f"{base_path}/{'crime_rates_regimes.csv'}"
    df = pd.read_csv(file, encoding="latin-1")
    df['CVEGEO'] = df['CVEGEO'].astype(str).str.zfill(5)
    return df

def w_contiguity():
    base_path = f"data/interim"
    file = f"{base_path}/{'adj_combined.csv'}"
    df = pd.read_csv(file, index_col=0)
    return df

def w_density():
    base_path = f"data/interim"
    file = f"{base_path}/{'W_density_contiguity.csv'}"
    df = pd.read_csv(file, encoding="latin-1")
    return df

def w_gravity():
    base_path = f"data/interim"
    file = f"{base_path}/{'gravity_matrix_2025.csv'}"
    df = pd.read_csv(file, encoding="latin-1")
    return df
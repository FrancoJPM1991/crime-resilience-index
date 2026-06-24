import pandas as pd
from src.config import DATA_INTERIM

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
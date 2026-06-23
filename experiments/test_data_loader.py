import pandas as pd
from src.data_loader import *

df1 = criminal_activity()
df2 = crime_rates()
df3 = w_contiguity()
print(df1.head())
print(df2.head())
print(df3.head())
print("Data loader functions loaded successfully.")
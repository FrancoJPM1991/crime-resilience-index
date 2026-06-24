from src.models.validation import validate
import pandas as pd

predictions_df = pd.read_csv("results/predictions_benchmark.csv")
validate(predictions_df)
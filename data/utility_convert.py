import pandas as pd
import os

parquet_files = [
    'cv_probabilities.parquet',
    'patients.parquet',
    'shap_oof.parquet',
    'df_cleaned.parquet'
]

for file in parquet_files:
    if os.path.exists(file):
        df = pd.read_parquet(file)
        csv_filename = file.replace('.parquet', '.csv')
        df.to_csv(csv_filename, index=False)
        print(csv_filename + " done")
    else:
        print("Error.\n")

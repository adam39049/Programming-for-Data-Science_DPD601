# src/data_loader.py

import pandas as pd


def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=";")
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    return df


def display_info(df):
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nTarget variable distribution:")
    print(df["Target"].value_counts())

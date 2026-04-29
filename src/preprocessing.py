# src/preprocessing.py
# القسم 2 — تنظيف البيانات      → preprocessing.py


import pandas as pd


def clean_dataset(df):
    # Check for null values
    print("\nNull values per column:")
    is_null = df.isnull().sum()
    null_columns = is_null[is_null > 0]

    if not null_columns.empty:
        for col in null_columns.index:
            percentage = (null_columns[col] / len(df)) * 100
            print(
                f"Column '{col}': {null_columns[col]} null values ({percentage:.2f}%)")
    else:
        print("No null values found in any column.")

    # Check for duplicated rows
    print("\nDuplicated rows:")
    duplicated_rows = df.duplicated()
    if duplicated_rows.any():
        print(f"Number of duplicated rows: {duplicated_rows.sum()}")
        before_deletion = len(df)
        df = df.drop_duplicates()
        after_deletion = len(df)
        print(f"Rows before deletion: {before_deletion}")
        print(f"Rows after deletion : {after_deletion}")
        print(f"Deleted             : {before_deletion - after_deletion} rows")
    else:
        print("No duplicated rows found.")

    return df

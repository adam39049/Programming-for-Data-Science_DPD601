def clean_dataset(df):
    # Check for null values
    print("\nNull values per column:")
    is_null = df.isnull().sum()
    null_columns = is_null[is_null > 0]

    if not null_columns.empty:
        for col in null_columns.index:
            percentage = (null_columns[col] / len(df)) * 100
            print(
                f"Column '{col}': {null_columns[col]} nulls ({percentage:.2f}%)")

            # If more than 50% missing — drop the column
            if percentage > 50:
                df = df.drop(columns=[col])
                print(f"  -> Column '{col}' dropped (too many nulls)")

            # If numeric — fill with median
            elif df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
                print(f"  -> Filled with median = {df[col].median():.2f}")

            # If categorical — fill with mode
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"  -> Filled with mode = '{df[col].mode()[0]}'")
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

    # Save processed dataset
    import os
    # os.makedirs('data/processed', exist_ok=True)
    dff = df.copy()
    dff.to_csv('data/processed/processed_dataset.csv', index=False)
    print(f"\nCleaned dataset saved successfully to: data/processed/processed_dataset.csv")

    return df

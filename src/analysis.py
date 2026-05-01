

import numpy as np


def numpy_analysis(df):
    # 1. Admission grade statistics using NumPy
    grades = df["Admission grade"].to_numpy()
    print("\nAdmission grade statistics:")
    print(f"Mean: {np.mean(grades):.2f}")
    print(f"Std : {np.std(grades):.2f}")
    print(f"Min : {np.min(grades):.2f}")
    print(f"Max : {np.max(grades):.2f}")

    # 2. Logical mask — how many students are older than 25?
    ages = df["Age at enrollment"].to_numpy()
    older_students = ages[ages > 25]
    print(f"\nStudents older than 25: {len(older_students)}")

    # 3. Percentile analysis of enrollment age
    print(f"\n25th percentile of age: {np.percentile(ages, 25):.1f}")
    print(f"75th percentile of age: {np.percentile(ages, 75):.1f}")

    # 4. Dropout rate by age group
    older_mask = ages > 25
    older_df = df[older_mask]
    younger_df = df[~older_mask]

    dropout_rate_older = (older_df["Target"] ==
                          "Dropout").sum() / len(older_df) * 100
    dropout_rate_younger = (
        younger_df["Target"] == "Dropout").sum() / len(younger_df) * 100

    print(
        f"\nDropout rate for students older than 25: {dropout_rate_older:.1f}%")
    print(
        f"Dropout rate for students 25 or younger: {dropout_rate_younger:.1f}%")


def eda_analysis(df):
    print("\nAdmission grade by target:")
    print(df.groupby("Target")["Admission grade"].describe().round(2))

    print("\nFirst semester grade by target:")
    print(df.groupby("Target")[
          "Curricular units 1st sem (grade)"].describe().round(2))

    print("\nDebtor distribution by target:")
    print(df.groupby("Target")["Debtor"].value_counts(normalize=True).round(2))

    print("\nScholarship distribution by target:")
    print(df.groupby("Target")["Scholarship holder"].value_counts(
        normalize=True).round(2))

    print("\nAttendance type by target:")
    print(df.groupby("Daytime/evening attendance\t")
          ["Target"].value_counts(normalize=True).round(2))

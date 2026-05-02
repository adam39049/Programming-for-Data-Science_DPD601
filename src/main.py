import os
import sys

# This must run before importing local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402


from analysis import eda_analysis, numpy_analysis  # noqa: E402
from data_loader import display_info, load_dataset  # noqa: E402
from modeling import (evaluate_model, prepare_data,  # noqa: E402
                      split_data, train_decision_tree,
                      train_random_forest)
from preprocessing import clean_dataset  # noqa: E402
from visualization import generate_all_visualizations  # noqa: E402
from evaluation import compare_and_save  # noqa: E402


# 1. Load Dataset

file_path = "data/raw/dataset.csv"
df = load_dataset(file_path)
display_info(df)
print("_" * 50)

# 2. Data Cleaning

df = clean_dataset(df)
print("_" * 50)


# 3. NumPy Analysis

numpy_analysis(df)
print("_" * 50)


# 4. EDA Analysis

eda_analysis(df)
print("_" * 50)


# 5. Visualization

generate_all_visualizations(df)
print("_" * 50)


# 6. Machine Learning
X, y = prepare_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
dt_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
dt_accuracy = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
rf_accuracy = evaluate_model(rf_model, X_test, y_test, "Random Forest")
print("_" * 50)

# 7. Compare and Save
compare_and_save(dt_model, rf_model, dt_accuracy, rf_accuracy)

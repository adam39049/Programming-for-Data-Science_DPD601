# src/main.py

from modeling import prepare_data, split_data, train_decision_tree, train_random_forest, evaluate_model
from visualization import generate_all_visualizations
from analysis import numpy_analysis, eda_analysis
from preprocessing import clean_dataset
from data_loader import load_dataset, display_info
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ========================================
# 1. Load Dataset
# ========================================
file_path = "data/raw/dataset.csv"
df = load_dataset(file_path)
display_info(df)

# ========================================
# 2. Data Cleaning
# ========================================
df = clean_dataset(df)

# ========================================
# 3. NumPy Analysis
# ========================================
numpy_analysis(df)

# ========================================
# 4. EDA Analysis
# ========================================
eda_analysis(df)

# ========================================
# 5. Visualization
# ========================================
generate_all_visualizations(df)

# ========================================
# 6. Machine Learning
# ========================================

# Prepare data
X, y = prepare_data(df)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Train models
dt_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate models
dt_accuracy = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
rf_accuracy = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Compare models
print("\n── Model Comparison ──")
print(f"Decision Tree accuracy : {dt_accuracy:.4f}")
print(f"Random Forest accuracy : {rf_accuracy:.4f}")

if rf_accuracy > dt_accuracy:
    print("Best model: Random Forest")
else:
    print("Best model: Decision Tree")

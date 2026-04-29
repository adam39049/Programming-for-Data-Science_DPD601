# src/main.py

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

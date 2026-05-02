# Academic Performance Analysis and Early Prediction of Student Dropout

## Student Information
- **Name:** Adam
- **ID:** 192566
- **Course:** DPD601 — Programming for Data Science
- **Instructor:** Dr. Doaa Kordab

---

## Project Title
Academic Performance Analysis and Early Prediction of Student Dropout

---

## Project Description
This project analyzes a real academic dataset of 4,424 students to understand
the factors that lead to dropout, continued enrollment, or graduation.
It applies data preprocessing, exploratory data analysis, data visualization,
and machine learning classification models to predict student academic status.

---

## Libraries and Tools Used
|     Library  |                 Purpose                  |
|--------------|------------------------------------------|
| Python       | Core programming language                |
| NumPy        | Numerical operations and logical masking |
| Pandas       | Data loading, cleaning, and analysis     |
| Matplotlib   | Data visualization                       |
| Seaborn      | Statistical visualizations               |
| Scikit-Learn | Machine learning models                  |

---

## Project Structure

project/
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── processed_dataset.csv
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── visualization.py
│   └── modeling.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── report/
│   └── final_report.docx
└── README.md

---

## Steps to Run the Project
1. Install required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Make sure the dataset is in place:

data/raw/dataset.csv

3. Run the main file from the project root:
```bash
python src/main.py
```

4. Outputs will be saved in:
- `outputs/figures/` — All visualizations
- `data/processed/` — Cleaned dataset

---
## Github

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adam39049/Programming-for-Data-Science_DPD601.git
   cd Programming-for-Data-Science_DPD601
   ```

2. **Install required libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the main file from the project root:**
```bash
python src/main.py
   

## Project Files Description
| File               |         Description                    |
|--------------------|----------------------------------------|
| `main.py`          | Entry point — runs the full pipeline   |
| `data_loader.py`   | Loads and displays dataset information |
| `preprocessing.py` | Cleans data and handles missing values |
| `analysis.py`      | NumPy operations and Pandas EDA        |
| `visualization.py` | Generates all 6 visualizations         |
| `modeling.py`      | Trains and evaluates ML models         |

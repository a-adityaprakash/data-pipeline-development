# Task 1: ETL Pipeline with Pandas and Scikit-Learn

## 📌 Objective
The objective of this task is to design and implement a professional **ETL (Extract, Transform, Load) pipeline** using **pandas** and **scikit-learn**. This work demonstrates essential data engineering skills required for preparing raw datasets into a structured and machine learning-ready format.

---

## 📖 Overview
The ETL pipeline consists of three primary stages:

1. **Data Extraction**  
   - Load raw data from CSV files using `pandas`.

2. **Data Preprocessing & Transformation**  
   - Handle missing values with `SimpleImputer`.  
   - Apply **scaling** to numeric features (`StandardScaler`).  
   - Apply **one-hot encoding** to categorical features (`OneHotEncoder`).  
   - Ensure the dataset is clean, consistent, and machine learning-ready.  

3. **Data Loading**  
   - Save the transformed datasets into structured CSV files for downstream tasks.  

---

## 🛠️ Tools & Libraries
- **Python 3.9+**
- **pandas** – for data extraction and manipulation.
- **scikit-learn** – for preprocessing, transformation, and pipeline management.
- **os** – for file management and structured data storage.

---

## 🚀 Usage Instructions
1. Place your dataset in the same directory as the script and rename it to `sample_data.csv`  
   *(The dataset must include a column named `target` as the prediction label).*

2. Run the ETL pipeline:
   ```bash
   python etl_pipeline.py
   
3. Processed datasets will be stored in the processed_data/ directory:

X_train.csv

X_test.csv

y_train.csv

y_test.csv

📂 Output Example
processed_data/
 ├── X_train.csv
 ├── X_test.csv
 ├── y_train.csv
 └── y_test.csv

"""
ETL Pipeline Script
-------------------
This script demonstrates an end-to-end ETL (Extract, Transform, Load) pipeline
using pandas and scikit-learn. It covers data preprocessing, transformation, 
and loading into a clean dataset ready for downstream tasks such as machine 
learning or analysis.

Author: [Your Name]
Internship Project: Data Engineering & Machine Learning
Date: [YYYY-MM-DD]
"""

# ==============================
# Imports
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os


# ==============================
# Step 1: Data Extraction
# ==============================
def extract_data(file_path: str) -> pd.DataFrame:
    """
    Extract raw data from CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    data = pd.read_csv(file_path)
    print("[INFO] Data extracted successfully.")
    return data


# ==============================
# Step 2: Data Preprocessing & Transformation
# ==============================
def transform_data(df: pd.DataFrame):
    """
    Clean, preprocess, and transform data using scikit-learn pipeline.
    """
    # Separate features and target (assuming `target` column exists)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Numeric preprocessing: impute missing values + scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing: impute missing + one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Apply transformations
    X_transformed = preprocessor.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )

    print("[INFO] Data transformation & preprocessing completed.")
    return X_train, X_test, y_train, y_test


# ==============================
# Step 3: Data Loading
# ==============================
def load_data(X_train, X_test, y_train, y_test, output_dir="processed_data"):
    """
    Save the transformed datasets to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train.toarray() if hasattr(X_train, "toarray") else X_train).to_csv(
        f"{output_dir}/X_train.csv", index=False
    )
    pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test).to_csv(
        f"{output_dir}/X_test.csv", index=False
    )
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"[INFO] Processed data saved in directory: {output_dir}")


# ==============================
# Main Function
# ==============================
if __name__ == "__main__":
    # Example CSV file path (replace with your own dataset)
    input_file = "sample_data.csv"

    # Run ETL pipeline
    df_raw = extract_data(input_file)
    X_train, X_test, y_train, y_test = transform_data(df_raw)
    load_data(X_train, X_test, y_train, y_test)
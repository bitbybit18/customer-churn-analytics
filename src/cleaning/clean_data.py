"""
clean_data.py
─────────────
Data cleaning and preprocessing for the IBM Telco Churn Dataset.
Handles missing values, data types, duplicates, and feature engineering.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path so we can import config
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

def load_raw_data():
    """Load the raw CSV dataset."""
    print("Loading raw data...")
    df = pd.read_csv(config.RAW_DATA_FILE)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def inspect_data(df):
    """Print a full inspection report of the dataset."""
    print("\n===== DATA INSPECTION =====")
    print(f"Shape        : {df.shape}")
    print(f"Duplicates   : {df.duplicated().sum()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nChurn Distribution:\n{df['Churn'].value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")

def clean_data(df):
    """
    Apply all cleaning steps and return a clean DataFrame.
    Each step is documented with the business reason.
    """

    print("\n===== CLEANING DATA =====")

    # ── Step 1: Remove duplicates ─────────────────────────────────────────────
    # Duplicate rows corrupt analysis and model training
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed : {before - len(df)}")

    # ── Step 2: Fix TotalCharges column ───────────────────────────────────────
    # TotalCharges is loaded as string because some values are blank spaces
    # These blanks belong to new customers (tenure = 0) who have no charges yet
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill nulls with 0 — new customers have not been charged yet
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print(f"TotalCharges nulls fixed")

    # ── Step 3: Convert SeniorCitizen from int to readable string ─────────────
    # Original is 0/1 — converting to No/Yes matches all other binary columns
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    print(f"SeniorCitizen converted to Yes/No")

    # ── Step 4: Strip whitespace from all string columns ─────────────────────
    # Hidden spaces cause GROUP BY mismatches in SQL and ML encoding errors
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    print(f"Whitespace stripped from {len(str_cols)} string columns")

    # ── Step 5: Standardize Churn column to 0/1 integer ──────────────────────
    # ML models need numeric target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print(f"Churn encoded: Yes=1, No=0")

    # ── Step 6: Feature Engineering ───────────────────────────────────────────
    # Create new features that help the model and SQL analysis

    # Tenure groups — useful for segmentation and Power BI slicers
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-12 months', '13-24 months', '25-48 months',
                '49-60 months', '61-72 months']
    )

    # Monthly charge tier — useful for revenue analysis
    df['charge_tier'] = pd.cut(
        df['MonthlyCharges'],
        bins=[0, 35, 65, 95, 120],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    print(f"Feature engineering complete: tenure_group, charge_tier added")

    return df

def save_clean_data(df):
    """Save the cleaned dataset to the processed folder."""
    config.PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_DATA_FILE, index=False)
    print(f"\nClean data saved to: {config.CLEAN_DATA_FILE}")
    print(f"Final shape: {df.shape}")

def main():
    df = load_raw_data()
    inspect_data(df)
    df = clean_data(df)
    save_clean_data(df)
    print("\nData cleaning complete!")

if __name__ == '__main__':
    main()

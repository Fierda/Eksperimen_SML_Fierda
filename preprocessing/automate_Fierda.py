import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import kaggle
except ImportError:
    print("[INFO] Installing kaggle...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    import kaggle

# ---------------------------------------------------------------------------
# 1. DOWNLOAD DATASET
# ---------------------------------------------------------------------------
def download_dataset(output_dir):
    raw_path = os.path.join(output_dir, "creditcard.csv")

    if os.path.exists(raw_path):
        print(f"[INFO] Dataset already exists at: {raw_path}")
        return raw_path

    print("[INFO] Downloading dataset from Kaggle...")
    os.makedirs(output_dir, exist_ok=True)

    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        print("[ERROR] KAGGLE_USERNAME and/or KAGGLE_KEY environment variable not found.")
        print("[HINT] Set the following environment variables before running the script:")
        print("         export KAGGLE_USERNAME=<your_kaggle_username>")
        print("         export KAGGLE_KEY=<kaggle_api_key>")
        print("         (Or add to GitHub Secrets if running via CI)")
        sys.exit(1)

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud", path=output_dir, unzip=True
        )
        print(f"[OK] Dataset successfully downloaded to: {raw_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download from Kaggle: {e}")
        sys.exit(1)

    return raw_path


# ---------------------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------------------
def load_data(filepath):
    print(f"\n[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Dtypes      :\n{df.dtypes.value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 3. EXPLORATORY CHECKS (automated summary)
# ---------------------------------------------------------------------------
def exploratory_checks(df):
    print("\n========== EXPLORATORY CHECKS ==========")
    print(f"  Missing values total : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows       : {df.duplicated().sum()}")
    print(f"  Class distribution   :\n{df['Class'].value_counts().to_string()}")
    fraud_pct = df['Class'].mean() * 100
    print(f"  Fraud percentage     : {fraud_pct:.4f}%")
    print(f"\n  Descriptive statistics (Amount):")
    print(f"    Mean   : {df['Amount'].mean():.2f}")
    print(f"    Median : {df['Amount'].median():.2f}")
    print(f"    Max    : {df['Amount'].max():.2f}")
    print(f"    Std    : {df['Amount'].std():.2f}")
    print("=" * 42)


# ---------------------------------------------------------------------------
# 4. PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess(df):
    print("\n[INFO] Starting preprocessing...")
    df_clean = df.copy()

    # a. Handle missing values
    n_missing = df_clean.isnull().sum().sum()
    if n_missing > 0:
        df_clean = df_clean.dropna()
        print(f"  [CLEAN] Dropped {n_missing} missing values → shape: {df_clean.shape}")
    else:
        print("  [CLEAN] No missing values found.")

    # b. Remove duplicates
    n_dupes = df_clean.duplicated().sum()
    if n_dupes > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"  [CLEAN] Dropped {n_dupes} duplicate rows → shape: {df_clean.shape}")
    else:
        print("  [CLEAN] No duplicate rows found.")

    # c. Feature scaling (StandardScaler) on Amount and Time
    scaler = StandardScaler()
    df_clean['Amount_scaled'] = scaler.fit_transform(df_clean[['Amount']])
    df_clean['Time_scaled'] = scaler.fit_transform(df_clean[['Time']])
    df_clean = df_clean.drop(['Amount', 'Time'], axis=1)
    print("  [SCALE] 'Amount' and 'Time' columns scaled and replaced.")

    # d. Undersampling majority class
    fraud = df_clean[df_clean['Class'] == 1]
    normal = df_clean[df_clean['Class'] == 0]
    n_fraud = len(fraud)
    n_sample = min(n_fraud * 3, len(normal))  # 3:1 ratio or all normal if less

    normal_undersampled = normal.sample(n=n_sample, random_state=42)
    df_balanced = pd.concat([normal_undersampled, fraud], axis=0)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  [BALANCE] Undersampling complete:")
    print(f"    Normal : {len(normal_undersampled)}")
    print(f"    Fraud  : {n_fraud}")
    print(f"    Total  : {len(df_balanced)}")

    return df_balanced


# ---------------------------------------------------------------------------
# 5. SAVE PREPROCESSED DATA
# ---------------------------------------------------------------------------
def save_preprocessed(df, output_path):
    """Save the processed dataset to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Preprocessed dataset saved to: {output_path}")
    print(f"     Final shape: {df.shape}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "creditcard_raw")
    output_path = os.path.join(script_dir, "creditcard_preprocessed.csv")

    print("=" * 60)
    print("  AUTOMATED PREPROCESSING PIPELINE")
    print("  Dataset: Credit Card Fraud Detection (Kaggle)")
    print("  Student: Fierda")
    print("=" * 60)

    raw_path = download_dataset(raw_dir)

    df = load_data(raw_path)

    exploratory_checks(df)

    df_clean = preprocess(df)

    save_preprocessed(df_clean, output_path)

    print("\n[DONE] Preprocessing pipeline finished!")
    return df_clean


if __name__ == "__main__":
    main()

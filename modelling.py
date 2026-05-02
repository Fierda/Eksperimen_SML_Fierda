import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "FraudDetection_Basic"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "creditcard_preprocessed.csv")

if not os.path.exists(DATA_PATH):
    ALT_PATH = os.path.join(SCRIPT_DIR, "preprocessing", "creditcard_preprocessed.csv")
    if os.path.exists(ALT_PATH):
        DATA_PATH = ALT_PATH


def train_basic():
    # Setup MLflow
    if TRACKING_URI != "mlruns":
        mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable autolog — will automatically log params, metrics, model
    mlflow.sklearn.autolog()

    # 1. Load data
    print(f"[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 2. Train/test split (stratified for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. Train model (without tuning — default hyperparameters)
    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("\n[RESULT] Classification Report:")
        print(report)

        print("[OK] Autolog has recorded params, metrics, and model automatically.")
        print(f"[OK] Check MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    train_basic()

import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
EXPERIMENT_NAME = "FraudDetection_Tuning_Advanced"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "creditcard_preprocessed.csv")

if not os.path.exists(DATA_PATH):
    ALT_PATH = os.path.join(SCRIPT_DIR, "preprocessing", "creditcard_preprocessed.csv")
    if os.path.exists(ALT_PATH):
        DATA_PATH = ALT_PATH


# ---------------------------------------------------------------------------
# DAGSHUB SETUP (optional, for Advanced)
# ---------------------------------------------------------------------------
def setup_dagshub():
    """Setup DagsHub tracking if env vars are available."""
    if TRACKING_URI:
        if os.environ.get("CI") == "true":
            mlflow.set_tracking_uri(TRACKING_URI)
            print(f"[OK] CI Environment detected. Direct MLflow tracking URI: {TRACKING_URI}")
        elif "dagshub" in TRACKING_URI:
            try:
                import dagshub
                dagshub.init(
                    repo_owner=os.environ.get("DAGSHUB_OWNER", ""),
                    repo_name=os.environ.get("DAGSHUB_REPO", ""),
                    mlflow=True,
                )
                print(f"[OK] DagsHub tracking active: {TRACKING_URI}")
            except ImportError:
                mlflow.set_tracking_uri(TRACKING_URI)
                print(f"[OK] MLflow tracking URI fallback: {TRACKING_URI}")
        else:
            mlflow.set_tracking_uri(TRACKING_URI)
            print(f"[OK] MLflow tracking URI: {TRACKING_URI}")
    else:
        print("[INFO] Using local MLflow tracking (./mlruns)")


# ---------------------------------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------------------------------
def train_and_log():
    setup_dagshub()
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load data
    print(f"\n[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. Hyperparameter Tuning
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
    )

    print("\n[INFO] Starting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"  Best params: {grid_search.best_params_}")

    # 4. Manual Logging to MLflow
    with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):

        # --- A. Log Parameters (identical to autolog) ---
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("class_weight", "balanced")

        # --- B. Log Metrics (identical to autolog + additional) ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("training_score", best_model.score(X_train, y_train))

        print(f"\n  [METRICS]")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
        print(f"    ROC AUC   : {auc:.4f}")
        print(f"    Log Loss  : {logloss:.4f}")

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
            ax=ax1,
        )
        ax1.set_title("Confusion Matrix - Fraud Detection")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        fig1.tight_layout()
        cm_path = "confusion_matrix.png"
        fig1.savefig(cm_path, dpi=150)
        plt.close(fig1)
        mlflow.log_artifact(cm_path)
        print(f"  [ARTIFACT] Confusion matrix saved & logged.")

        # --- Feature Importance Plot ---
        feat_importances = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=True)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        feat_importances.tail(15).plot(kind="barh", color="steelblue", ax=ax2)
        ax2.set_title("Top 15 Feature Importances")
        ax2.set_xlabel("Importance")
        fig2.tight_layout()
        fi_path = "feature_importance.png"
        fig2.savefig(fi_path, dpi=150)
        plt.close(fig2)
        mlflow.log_artifact(fi_path)
        print(f"  [ARTIFACT] Feature importance plot saved & logged.")

        # --- Classification Report (JSON) ---
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(report_path)
        print(f"  [ARTIFACT] Classification report JSON saved & logged.")

        # --- Log Model ---
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="fraud_rf_model",
            registered_model_name="FraudModel",
        )
        print(f"  [MODEL] Model logged & registered as 'FraudModel'.")

        # --- Log Tags ---
        mlflow.set_tag("developer", "Fierda")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", "Credit Card Fraud Detection")
        mlflow.set_tag("tuning_method", "GridSearchCV")

    for f in [cm_path, fi_path, report_path]:
        if os.path.exists(f):
            os.remove(f)

    print("\n[DONE] Training complete! All metrics and artifacts have been logged.")
    print("[INFO] Run 'mlflow ui --port 5000' to view the results in the browser.")


if __name__ == "__main__":
    train_and_log()

"""
Advanced modelling script with hyperparameter tuning and manual MLflow logging.
Trains an optimized XGBoost classifier on the credit card fraud dataset.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple
import json

import dagshub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tempfile
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "creditcardfraud_preprocessing.csv"
)
TEST_SIZE = 0.2
RANDOM_STATE = 42
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
PROXY_COLS: List[str] = ["transaction_id"]
CV_FOLDS = 5
CV_REPEATS = 3


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset and drop target plus obvious proxy columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    if "is_fraud" not in df.columns:
        raise KeyError("Column 'is_fraud' not found in dataset")

    df = df.copy()
    y = df.pop("is_fraud")

    # Drop proxy/id columns if present to avoid leakage
    for col in PROXY_COLS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Safety: drop any stray target column copies
    if "is_fraud" in df.columns:
        df.drop(columns="is_fraud", inplace=True)

    X = df
    return X, y


def plot_confusion_matrix(cm: np.ndarray, labels: Tuple[str, str], path: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ap: float, path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR Curve (AUPRC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str], path: str, top_n: int = 20) -> None:
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=vals, y=names, orient="h")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance (Top %d)" % min(top_n, len(feature_names)))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_cv(best_params: dict, X: pd.DataFrame, y: pd.Series, path_prefix: str) -> dict:
    cv_estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cv_estimator.set_params(**best_params)

    rskf = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )

    f1_scores = cross_val_score(cv_estimator, X, y, cv=rskf, scoring="f1", n_jobs=-1)
    precision_scores = cross_val_score(
        cv_estimator, X, y, cv=rskf, scoring="precision", n_jobs=-1
    )
    recall_scores = cross_val_score(
        cv_estimator, X, y, cv=rskf, scoring="recall", n_jobs=-1
    )
    ap_scores = cross_val_score(
        cv_estimator, X, y, cv=rskf, scoring="average_precision", n_jobs=-1
    )

    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    sns.histplot(f1_scores, bins=10, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title("CV F1")
    sns.histplot(precision_scores, bins=10, ax=axes[0, 1], kde=True)
    axes[0, 1].set_title("CV Precision")
    sns.histplot(recall_scores, bins=10, ax=axes[1, 0], kde=True)
    axes[1, 0].set_title("CV Recall")
    sns.histplot(ap_scores, bins=10, ax=axes[1, 1], kde=True)
    axes[1, 1].set_title("CV AUPRC")
    plt.tight_layout()
    hist_path = os.path.join(path_prefix, "cv_metrics_hist.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # Save raw scores
    raw_path = os.path.join(path_prefix, "cv_scores.json")
    with open(raw_path, "w") as f:
        json.dump(
            {
                "f1": f1_scores.tolist(),
                "precision": precision_scores.tolist(),
                "recall": recall_scores.tolist(),
                "auprc": ap_scores.tolist(),
            },
            f,
        )

    return {
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores, ddof=1)),
        "precision_mean": float(np.mean(precision_scores)),
        "precision_std": float(np.std(precision_scores, ddof=1)),
        "recall_mean": float(np.mean(recall_scores)),
        "recall_std": float(np.std(recall_scores, ddof=1)),
        "auprc_mean": float(np.mean(ap_scores)),
        "auprc_std": float(np.std(ap_scores, ddof=1)),
        "hist_path": hist_path,
        "raw_path": raw_path,
    }


def tune_and_log(X: pd.DataFrame, y: pd.Series) -> None:
    # Set MLflow tracking URI from environment variable or DagsHub
    # Format: https://dagshub.com/<owner>/<repo>.mlflow
    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/Dekhsa/Workflow-CI.mlflow"
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Pipeline to ensure transformations are fit only on training data
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__n_estimators": [100, 200],
        "model__scale_pos_weight": [1.0, 5.0],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name="Optimized_Model_Tuning"):
        # Log dataset artifacts and register datasets so they appear in UI
        try:
            mlflow.log_artifact(DATA_PATH, artifact_path="data")
            mlflow.log_dict(
                {
                    "rows": int(X.shape[0]),
                    "columns": int(X.shape[1]),
                    "features": list(X.columns),
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                },
                artifact_file="data/dataset_profile.json",
            )

            train_df = X_train.copy()
            train_df["is_fraud"] = y_train.to_numpy()
            test_df = X_test.copy()
            test_df["is_fraud"] = y_test.to_numpy()

            train_ds = mlflow.data.from_pandas(
                train_df, source=DATA_PATH, name="creditcard_train"
            )
            test_ds = mlflow.data.from_pandas(
                test_df, source=DATA_PATH, name="creditcard_test"
            )
            mlflow.log_input(train_ds, context="training")
            mlflow.log_input(test_ds, context="testing")
        except Exception:
            # Non-fatal if dataset logging fails
            pass
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auprc = average_precision_score(y_test, y_proba)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auprc", auprc)

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        pr_path = os.path.join(ARTIFACT_DIR, "precision_recall_curve.png")

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels=("No Fraud", "Fraud"), path=cm_path)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        plot_pr_curve(prec_curve, rec_curve, auprc, path=pr_path)

        # Log feature importance visualization
        fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
        # Access the XGBClassifier inside the pipeline
        xgb_model: XGBClassifier = best_model.named_steps["model"]
        plot_feature_importance(xgb_model.feature_importances_, list(X.columns), fi_path)

        # Log artifacts
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(fi_path)

        # Log the trained pipeline as an MLflow model (creates MLmodel, model.pkl, env spec)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Save to a temporary directory to avoid collisions across runs, then upload folder
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_dir = os.path.join(tmpdir, "model")
            mlflow.sklearn.save_model(best_model, path=local_model_dir)
            mlflow.log_artifacts(local_model_dir, artifact_path="model")

        print("Best Params:", grid.best_params_)
        print(
            f"F1: {f1:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, AUPRC: {auprc:.4f}"
        )

        # Cross-validation to reduce variance risk
        cv_results = evaluate_cv(grid.best_params_, X, y, ARTIFACT_DIR)
        mlflow.log_metric("cv_f1_mean", cv_results["f1_mean"])
        mlflow.log_metric("cv_f1_std", cv_results["f1_std"])
        mlflow.log_metric("cv_precision_mean", cv_results["precision_mean"])
        mlflow.log_metric("cv_precision_std", cv_results["precision_std"])
        mlflow.log_metric("cv_recall_mean", cv_results["recall_mean"])
        mlflow.log_metric("cv_recall_std", cv_results["recall_std"])
        mlflow.log_metric("cv_auprc_mean", cv_results["auprc_mean"])
        mlflow.log_metric("cv_auprc_std", cv_results["auprc_std"])

        mlflow.log_artifact(cv_results["hist_path"])
        mlflow.log_artifact(cv_results["raw_path"])

        print(
            f"CV F1 mean={cv_results['f1_mean']:.3f} std={cv_results['f1_std']:.3f}; "
            f"CV Precision mean={cv_results['precision_mean']:.3f} std={cv_results['precision_std']:.3f}; "
            f"CV Recall mean={cv_results['recall_mean']:.3f} std={cv_results['recall_std']:.3f}; "
            f"CV AUPRC mean={cv_results['auprc_mean']:.3f} std={cv_results['auprc_std']:.3f}"
        )


def main() -> None:
    try:
        X, y = load_data(DATA_PATH)
        tune_and_log(X, y)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Error during model tuning: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()


def main() -> None:
    try:
        X, y = load_data(DATA_PATH)
        tune_and_log(X, y)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Error during model tuning: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

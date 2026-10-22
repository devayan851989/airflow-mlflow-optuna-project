#!/usr/bin/env python3
"""Compare baseline regressors and track their runs in MLflow."""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from airflow import DAG
from airflow.decorators import task


log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "housing_price_prediction"
REGISTERED_MODEL = "HousingPricePredictor"
FEATURE_COLS = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "house_age",
    "distance_km",
    "garage_spots",
    "bed_bath_ratio",
    "is_new_house",
]
TARGET_COL = "price_usd"

default_args = {
    "owner": "mlops-learner",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)


def _get_data():
    data_path = PROCESSED_DIR / "housing_clean.parquet"
    if not data_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run scripts/prepare_data.py first.")
    frame = pd.read_parquet(data_path)
    missing = sorted(set(FEATURE_COLS + [TARGET_COL]) - set(frame.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")
    return train_test_split(
        frame[FEATURE_COLS], frame[TARGET_COL], test_size=0.2, random_state=42
    )


def _log_regression_run(model, run_name: str, params: dict, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=run_name) as run:
        cv_r2 = float(cross_val_score(model, X_train, y_train, cv=3, scoring="r2").mean())
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "cv_r2": round(cv_r2, 4),
            "mae": round(mean_absolute_error(y_test, predictions), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 2),
            "r2": round(r2_score(y_test, predictions), 4),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=REGISTERED_MODEL,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        return {"run_id": run.info.run_id, "model": run_name, **metrics}


@task(task_id="run_ridge_experiment")
def run_ridge_experiment():
    """Fit a scaled Ridge baseline and log it to MLflow."""
    _setup_mlflow()
    X_train, X_test, y_train, y_test = _get_data()
    return _log_regression_run(
        make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "Ridge_Regression",
        {"alpha": 1.0, "model_type": "ridge"}, X_train, X_test, y_train, y_test,
    )


@task(task_id="run_rf_experiment")
def run_rf_experiment():
    """Fit a Random Forest baseline and log model importance as an artifact."""
    _setup_mlflow()
    X_train, X_test, y_train, y_test = _get_data()
    params = {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_split": 4,
        "model_type": "random_forest",
    }
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        n_jobs=-1,
        random_state=42,
    )
    result = _log_regression_run(model, "Random_Forest", params, X_train, X_test, y_train, y_test)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.round(4)))
    importance_path = MODELS_DIR / "feature_importance.json"
    importance_path.write_text(
        json.dumps(dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)), indent=2) + "\n"
    )
    mlflow.tracking.MlflowClient().log_artifact(
        result["run_id"], str(importance_path), artifact_path="feature_importance"
    )
    return result


@task(task_id="run_gb_experiment")
def run_gb_experiment():
    """Fit a Gradient Boosting baseline and log it to MLflow."""
    _setup_mlflow()
    X_train, X_test, y_train, y_test = _get_data()
    params = {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 5,
        "subsample": 0.85,
        "model_type": "gradient_boosting",
    }
    model = GradientBoostingRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        random_state=42,
    )
    return _log_regression_run(model, "GradientBoosting", params, X_train, X_test, y_train, y_test)


@task(task_id="compare_and_promote")
def compare_and_promote(ridge: dict, rf: dict, gb: dict):
    """Select by cross-validation and transition its registered version to Staging."""
    _setup_mlflow()
    candidates = [ridge, rf, gb]
    best = max(candidates, key=lambda result: result["cv_r2"])
    for candidate in candidates:
        log.info("%s %-20s CV R²=%.4f | test R²=%.4f | RMSE=$%.0f",
                 "BEST" if candidate == best else "    ", candidate["model"],
                 candidate["cv_r2"], candidate["r2"], candidate["rmse"])

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    matching = [version for version in versions if version.run_id == best["run_id"]]
    if matching:
        version = max(matching, key=lambda item: int(item.version))
        client.transition_model_version_stage(
            name=REGISTERED_MODEL, version=version.version, stage="Staging"
        )
        log.info("Promoted %s version %s to Staging", REGISTERED_MODEL, version.version)
    else:
        log.warning("No registered version found for selected MLflow run %s", best["run_id"])
    return best


with DAG(
    dag_id="tutorial_mlflow_tracking",
    description="Compare housing-price baselines with MLflow tracking and model registry",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "mlflow", "model-tracking"],
) as dag:
    ridge_result = run_ridge_experiment()
    rf_result = run_rf_experiment()
    gb_result = run_gb_experiment()
    compare_and_promote(ridge_result, rf_result, gb_result)
#!/usr/bin/env python3
"""Tune tree-based regressors with Optuna and track results in MLflow."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from airflow import DAG
from airflow.decorators import task


log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "housing_hyperparameter_tuning"
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "30"))
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


def _load_data():
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


def _cross_validated_r2(trial: optuna.Trial, model, X_train, y_train) -> float:
    """Evaluate folds one by one so Optuna can prune weak candidates early."""
    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(X_train)):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
        predictions = fold_model.predict(X_train.iloc[validation_indices])
        scores.append(r2_score(y_train.iloc[validation_indices], predictions))
        trial.report(float(np.mean(scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


def _run_study(study_name: str, objective, X_train, y_train):
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study


def _log_tuned_model(model, run_name: str, best_params: dict, cv_r2: float, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_r2": round(cv_r2, 4),
            "test_r2": round(test_r2, 4),
            "n_trials": N_TRIALS,
        })
        mlflow.sklearn.log_model(
            model,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id
    return test_r2, run_id


@task(task_id="tune_random_forest")
def tune_random_forest():
    """Search Random Forest parameters, then fit and log the best candidate."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    X_train, X_test, y_train, y_test = _load_data()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
        return _cross_validated_r2(trial, model, X_train, y_train)

    study = _run_study("rf_tuning", objective, X_train, y_train)
    best_params = study.best_params
    final_model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
    test_r2, run_id = _log_tuned_model(
        final_model, "RF_Optuna_Best", best_params, study.best_value,
        X_train, y_train, X_test, y_test,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "rf_tuned.pkl")
    return {
        "model": "rf_tuned",
        "cv_r2": round(study.best_value, 4),
        "test_r2": round(test_r2, 4),
        "run_id": run_id,
        "best_params": best_params,
    }


@task(task_id="tune_gradient_boosting")
def tune_gradient_boosting():
    """Search Gradient Boosting parameters, then fit and log the best candidate."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    X_train, X_test, y_train, y_test = _load_data()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
        model = GradientBoostingRegressor(**params, random_state=42)
        return _cross_validated_r2(trial, model, X_train, y_train)

    study = _run_study("gb_tuning", objective, X_train, y_train)
    best_params = study.best_params
    final_model = GradientBoostingRegressor(**best_params, random_state=42)
    test_r2, run_id = _log_tuned_model(
        final_model, "GB_Optuna_Best", best_params, study.best_value,
        X_train, y_train, X_test, y_test,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "gb_tuned.pkl")
    return {
        "model": "gb_tuned",
        "cv_r2": round(study.best_value, 4),
        "test_r2": round(test_r2, 4),
        "run_id": run_id,
        "best_params": best_params,
    }


@task(task_id="finalize_best_tuned_model")
def finalize_best_tuned_model(rf_result: dict, gb_result: dict):
    """Select by cross-validation and persist the held-out test results."""
    winner = max([rf_result, gb_result], key=lambda result: result["cv_r2"])
    runner_up = gb_result if winner is rf_result else rf_result
    source_path = MODELS_DIR / f"{winner['model']}.pkl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Expected tuned model artifact not found: {source_path}")
    joblib.dump(joblib.load(source_path), MODELS_DIR / "best_tuned_model.pkl")

    summary = {
        "winner": winner,
        "runner_up": runner_up,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "cv_r2",
    }
    (MODELS_DIR / "tuning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Selected %s (test R²=%.4f)", winner["model"], winner["test_r2"])
    return summary


with DAG(
    dag_id="tutorial_hyperparameter_tuning",
    description="Tune housing-price regressors with Optuna and track results in MLflow",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "optuna", "hyperparameter-tuning", "mlflow"],
) as dag:
    rf_result = tune_random_forest()
    gb_result = tune_gradient_boosting()
    finalize_best_tuned_model(rf_result, gb_result)
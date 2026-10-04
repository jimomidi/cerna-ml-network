"""
Train and benchmark regression models that predict miRNA-mRNA interaction
strength (Pearson correlation between expression profiles) from paired
expression features, and persist the artifacts needed by the downstream
interpretation and network-analysis scripts.

Usage
-----
    python train_models.py --group case
    python train_models.py --group control

Run once per sample group (see config.GROUP_A_NAME / config.GROUP_B_NAME).
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

import config
import visualization
from data_processing import load_group_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _candidate_models() -> Dict[str, object]:
    """Regression models benchmarked against the production RandomForest model."""
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        "RandomForest": RandomForestRegressor(**config.RANDOM_FOREST_PARAMS),
    }

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
    except ImportError:
        logger.warning("lightgbm is not installed; skipping LightGBM in the model comparison.")

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
    except ImportError:
        logger.warning("xgboost is not installed; skipping XGBoost in the model comparison.")

    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute MSE, MAE and R2 for a fitted model on held-out data."""
    y_pred = model.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def train_and_compare_models(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Fit every candidate model and return its held-out evaluation metrics."""
    metrics = {}
    for name, model in _candidate_models().items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        metrics[name] = evaluate_model(model, X_test, y_test)
        logger.info("%s: %s", name, metrics[name])
    return metrics


def run_pipeline(group: str) -> None:
    """Run preprocessing, model benchmarking, and production-model training for one group."""
    if group == config.GROUP_A_NAME:
        mirna_path, mrna_path = config.GROUP_A_MIRNA_FILE, config.GROUP_A_MRNA_FILE
    elif group == config.GROUP_B_NAME:
        mirna_path, mrna_path = config.GROUP_B_MIRNA_FILE, config.GROUP_B_MRNA_FILE
    else:
        raise ValueError(f"Unknown group '{group}'. Expected one of "
                          f"'{config.GROUP_A_NAME}' or '{config.GROUP_B_NAME}'.")

    group_results_dir = config.RESULTS_DIR / group
    group_figures_dir = config.FIGURES_DIR / group
    group_results_dir.mkdir(parents=True, exist_ok=True)
    group_figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, pair_list, mirna_expr, mrna_expr = load_group_dataset(
        mirna_path, mrna_path, config.KNOWN_INTERACTIONS_FILE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    logger.info("Benchmarking candidate regression models")
    metrics = train_and_compare_models(X_train, y_train, X_test, y_test)
    pd.DataFrame(metrics).T.to_csv(group_results_dir / "model_comparison.csv")
    visualization.plot_model_comparison(metrics, group_figures_dir / "model_comparison.svg")

    logger.info("Training production model (RandomForestRegressor)")
    model = RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    final_metrics = evaluate_model(model, X_test, y_test)
    logger.info("Production model performance: %s", final_metrics)

    cv_scores = cross_val_score(model, X, y, cv=config.CROSS_VAL_FOLDS, scoring="r2")
    logger.info("Cross-validated R2: mean=%.4f std=%.4f", cv_scores.mean(), cv_scores.std())

    # Diagnostic plots.
    visualization.plot_true_vs_predicted(y_test, y_pred, group_figures_dir / "true_vs_predicted.svg")
    visualization.plot_score_distribution(y_test, y_pred, group_figures_dir / "score_distribution.svg")
    visualization.plot_residuals(y_test, y_pred, group_figures_dir / "residuals.svg")
    visualization.plot_error_by_expression_group(y_test, y_pred, save_path=group_figures_dir / "error_by_group.svg")

    # Persist artifacts needed by model_interpretation.py and network_analysis.py.
    joblib.dump(model, group_results_dir / "model.pkl")
    np.save(group_results_dir / "X_all.npy", X)
    np.save(group_results_dir / "y_all.npy", y)
    np.save(group_results_dir / "y_test.npy", y_test)
    np.save(group_results_dir / "y_pred.npy", y_pred)

    pairs_df = pd.DataFrame(pair_list, columns=["miRNA_ID", "gene_name"])
    with open(group_results_dir / "pairs.pkl", "wb") as f:
        pickle.dump(pairs_df, f)

    feature_names = list(mirna_expr.index) + list(mrna_expr.index)
    pd.Series(feature_names).to_csv(group_results_dir / "feature_names.csv", index=False, header=False)

    with open(group_results_dir / "metrics.txt", "w") as f:
        f.write(f"Held-out metrics: {final_metrics}\n")
        f.write(f"Cross-validated R2: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}\n")

    logger.info("Artifacts saved to %s", group_results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=[config.GROUP_A_NAME, config.GROUP_B_NAME],
        required=True,
        help="Sample group to process.",
    )
    args = parser.parse_args()
    run_pipeline(args.group)


if __name__ == "__main__":
    main()

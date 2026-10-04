"""
Interpret a trained interaction-strength model using SHAP (SHapley Additive
exPlanations) and built-in feature importances.

Requires artifacts produced by ``train_models.py`` for the requested group
(model.pkl, X_all.npy, feature_names.csv).

Usage
-----
    python model_interpretation.py --group case
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
from visualization import plot_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_group_artifacts(group: str):
    results_dir = config.RESULTS_DIR / group
    model = joblib.load(results_dir / "model.pkl")
    X = np.load(results_dir / "X_all.npy")
    feature_names = pd.read_csv(results_dir / "feature_names.csv", header=None)[0].tolist()
    feature_names = _resolve_feature_names(feature_names, X.shape[1])
    return model, X, feature_names


def _resolve_feature_names(feature_names: list, n_features: int) -> list:
    """Return a feature-name list guaranteed to match the model's input width.

    ``feature_names.csv`` records the miRNA/gene identifiers that passed the
    expression filter, for reference. The model's actual input columns are
    the concatenated per-sample expression values of one miRNA-gene pair, so
    the two lists only line up when the number of retained identifiers
    happens to equal 2 x (number of samples). If they do not match, this
    falls back to generic column labels so interpretation plots stay
    correctly aligned with the model instead of silently mislabeling
    features.
    """
    if len(feature_names) == n_features:
        return feature_names
    logger.warning(
        "feature_names.csv has %d entries but the model expects %d input "
        "features; falling back to generic column labels for interpretation "
        "plots. See README 'Known limitations' for details.",
        len(feature_names), n_features,
    )
    return [f"feature_{i}" for i in range(n_features)]


def filter_high_confidence_pairs(
    model, X: np.ndarray, score_threshold: float = config.SHAP_SCORE_THRESHOLD
) -> np.ndarray:
    """Restrict X to pairs the model scores above ``score_threshold``.

    SHAP interpretation is focused on high-confidence predicted interactions
    (predicted score above the network confidence threshold) rather than the
    full candidate set, keeping the explanation aligned with the pairs that
    actually enter the downstream network.
    """
    scores = model.predict(X)
    mask = scores > score_threshold
    logger.info(
        "Restricting SHAP analysis to %d/%d pairs with predicted score > %.2f",
        mask.sum(), len(mask), score_threshold,
    )
    if not mask.any():
        logger.warning(
            "No pairs exceed the SHAP score threshold (%.2f); using the full "
            "dataset instead.", score_threshold,
        )
        return X
    return X[mask]


def compute_shap_values(model, X: np.ndarray):
    """Compute SHAP values using a tree-path-dependent explainer.

    Falls back to the generic ``shap.Explainer`` if the model is not a
    supported tree-based estimator.
    """
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        return explainer.shap_values(X)
    except Exception:
        logger.info("Falling back to generic SHAP explainer.")
        explainer = shap.Explainer(model, X)
        return explainer(X)


def plot_shap_beeswarm(shap_values, X, feature_names, max_display: int, save_path: Path) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display, show=False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()


def plot_shap_violin(shap_values, X, feature_names, max_display: int, save_path: Path) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display,
                       plot_type="violin", show=False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()


def plot_shap_heatmap(model, X, feature_names, n_samples: int, save_path: Path) -> None:
    explainer = shap.Explainer(model, X, feature_names=feature_names)
    shap_values = explainer(X)
    shap.plots.heatmap(shap_values[:n_samples], max_display=20, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence(shap_values, X, feature_names, feature: str, save_path: Path) -> None:
    if feature not in feature_names:
        logger.warning("Feature '%s' not found; skipping SHAP dependence plot.", feature)
        return
    X_df = pd.DataFrame(X, columns=feature_names)
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_df, show=False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()


def run_pipeline(group: str) -> None:
    model, X, feature_names = load_group_artifacts(group)
    figures_dir = config.FIGURES_DIR / group

    X_high_confidence = filter_high_confidence_pairs(model, X)
    shap_values = compute_shap_values(model, X_high_confidence)

    plot_shap_beeswarm(shap_values, X_high_confidence, feature_names, config.SHAP_BEESWARM_MAX_DISPLAY,
                        figures_dir / "shap_beeswarm.svg")
    plot_shap_violin(shap_values, X_high_confidence, feature_names, config.SHAP_SUMMARY_MAX_DISPLAY,
                      figures_dir / "shap_violin.svg")
    plot_shap_heatmap(model, X_high_confidence, feature_names, n_samples=10,
                       save_path=figures_dir / "shap_heatmap.svg")

    if config.SHAP_DEPENDENCE_FEATURE:
        plot_shap_dependence(shap_values, X_high_confidence, feature_names, config.SHAP_DEPENDENCE_FEATURE,
                              figures_dir / "shap_dependence.svg")

    if hasattr(model, "feature_importances_"):
        plot_feature_importance(
            feature_names,
            model.feature_importances_,
            top_n=config.FEATURE_IMPORTANCE_TOP_N,
            save_path=figures_dir / "feature_importance.svg",
        )

    logger.info("Interpretation figures saved to %s", figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=[config.GROUP_A_NAME, config.GROUP_B_NAME],
        required=True,
        help="Sample group to interpret (must have been trained already).",
    )
    args = parser.parse_args()
    run_pipeline(args.group)


if __name__ == "__main__":
    main()

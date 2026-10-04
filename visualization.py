"""
Reusable plotting functions for model diagnostics and network summaries.

Every function optionally saves the generated figure to disk (as SVG, the
preferred vector format used throughout this project) when a ``save_path``
is given, and always returns the created matplotlib Figure for further
customization or inline display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FONT_FAMILY = "DejaVu Sans"


def _save(fig: plt.Figure, save_path: Optional[Path]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="svg", bbox_inches="tight")


def plot_true_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot of true vs. predicted interaction scores with a y=x reference line."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, color="#053061", alpha=0.5)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=1.5)
    ax.set_xlabel("True correlation", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted correlation", fontsize=14, fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid histograms of true and predicted score distributions."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.histplot(y_true, kde=True, color="#053061", label="True", stat="count", bins=50, alpha=0.65, ax=ax)
    sns.histplot(y_pred, kde=True, color="#67001f", label="Predicted", stat="count", bins=50, alpha=0.65, ax=ax)
    ax.set_xlabel("Correlation score", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of prediction residuals (true - predicted)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.histplot(residuals, kde=True, bins=100, color="#053061", alpha=0.6, ax=ax)
    ax.set_xlabel("Residual (true - predicted)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_error_by_expression_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_groups: int = 3,
    group_labels: Sequence[str] = ("Low", "Medium", "High"),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Boxplot comparing absolute and signed prediction error across target-value tertiles."""
    abs_error = np.abs(y_true - y_pred)
    signed_error = y_true - y_pred
    group = pd.qcut(y_true, q=n_groups, labels=group_labels)

    df = pd.concat(
        [
            pd.DataFrame({"Group": group, "Error": abs_error, "Error type": "Absolute"}),
            pd.DataFrame({"Group": group, "Error": signed_error, "Error type": "Signed"}),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"Absolute": "#67001f", "Signed": "#053061"}
    sns.boxplot(data=df, x="Group", y="Error", hue="Error type", palette=palette, showcaps=True, fliersize=0, ax=ax)
    ax.set_xlabel("Target-value group", fontsize=14, fontweight="bold")
    ax.set_ylabel("Prediction error", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_model_comparison(
    metrics: Mapping[str, Mapping[str, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal grouped bar chart comparing MSE, MAE and R2 across models.

    ``metrics`` maps model name -> {"mse": ..., "mae": ..., "r2": ...}.
    """
    models = list(metrics.keys())
    mse = [metrics[m]["mse"] for m in models]
    mae = [metrics[m]["mae"] for m in models]
    r2 = [metrics[m]["r2"] for m in models]

    index = np.arange(len(models))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(index - bar_width, mse, bar_width, label="MSE", color="#053061")
    ax.barh(index, mae, bar_width, label="MAE", color="#67001f")
    ax.barh(index + bar_width, r2, bar_width, label="R2", color="#888888")

    ax.set_yticks(index)
    ax.set_yticklabels(models, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    top_n: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-N most important features."""
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(top_n), importances[order][::-1], color="#053061")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=11)
    ax.set_xlabel("Feature importance", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_degree_distribution(
    degree_by_group: Mapping[str, pd.Series],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Log-log scatter plot of node-degree distributions for one or more groups.

    ``degree_by_group`` maps group label -> a Series/array of per-node degree
    values.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#053061", "#67001f", "#1b7837", "#762a83"]
    for color, (label, degrees) in zip(colors, degree_by_group.items()):
        counts = pd.Series(degrees).value_counts().sort_index()
        ax.scatter(counts.index, counts.values, s=60, alpha=0.7, color=color, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of nodes", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_selected_feature_degrees(
    degree_df: pd.DataFrame,
    feature_names: Sequence[str],
    color: str = "#053061",
    degree_column: str = "degree",
    name_column: str = "gene_name",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of node degree for a user-specified list of features of interest.

    ``degree_df`` is expected to contain at least ``name_column`` and
    ``degree_column``. Returns an empty figure if ``feature_names`` is empty.
    """
    subset = degree_df[degree_df[name_column].isin(feature_names)].sort_values(degree_column)

    fig, ax = plt.subplots(figsize=(5, 5))
    if not subset.empty:
        ax.bar(subset[name_column], subset[degree_column], color=color, width=0.6)
    ax.set_ylabel("Degree", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_threshold_sensitivity(
    thresholds: Sequence[float],
    counts_by_group: Mapping[str, Mapping[str, Sequence[int]]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart of network size (miRNA/mRNA/edge counts) across score thresholds.

    ``counts_by_group`` maps group label -> {"mirna": [...], "mrna": [...], "edges": [...]},
    each list aligned with ``thresholds``.
    """
    labels = [str(t) for t in thresholds]
    x = np.arange(len(labels))
    n_series = sum(len(v) for v in counts_by_group.values())
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    offset = -0.5 * (n_series - 1)
    for group_label, series in counts_by_group.items():
        for series_label, values in series.items():
            ax.bar(x + offset * width, values, width, label=f"{group_label} {series_label}")
            offset += 1

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Score threshold", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_interaction_heatmap(
    score_table: pd.DataFrame,
    index_col: str = "miRNA_ID",
    columns_col: str = "gene_name",
    values_col: str = "score",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of predicted interaction scores for a table of (miRNA, gene, score) rows."""
    matrix = score_table.pivot(index=index_col, columns=columns_col, values=values_col).fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.4), max(4, matrix.shape[0] * 0.3)))
    sns.heatmap(matrix, cmap="RdBu_r", mask=(matrix == 0), linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Predicted score"}, ax=ax)
    ax.set_xlabel(columns_col)
    ax.set_ylabel(index_col)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    _save(fig, save_path)
    return fig

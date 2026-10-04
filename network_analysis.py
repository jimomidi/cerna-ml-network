"""
Construct a ceRNA-style miRNA-mRNA regulatory network from predicted
interaction scores, evaluate its robustness across score thresholds, and
produce network visualizations and summary statistics.

Requires artifacts produced by ``train_models.py`` for the requested group
(model.pkl, X_all.npy, pairs.pkl).

Usage
-----
    python network_analysis.py --group case
    python network_analysis.py --group case --compare-with control
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

import config
import visualization
from data_processing import load_mrna_expression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_group_predictions(group: str) -> pd.DataFrame:
    """Load a group's trained model and candidate pairs, and score every pair."""
    results_dir = config.RESULTS_DIR / group
    model = joblib.load(results_dir / "model.pkl")
    X = np.load(results_dir / "X_all.npy")

    with open(results_dir / "pairs.pkl", "rb") as f:
        pairs = pickle.load(f)

    pairs = pairs.copy()
    pairs["score"] = model.predict(X)
    return pairs


def deduplicate_pair_scores(pairs: pd.DataFrame) -> pd.DataFrame:
    """Average the predicted score for any duplicated (miRNA, gene) pair."""
    return pairs.groupby(["miRNA_ID", "gene_name"], as_index=False)["score"].mean()


def compute_coexpression_matrix(mrna_expr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Compute a gene-by-gene Pearson correlation matrix and two-sided p-values.

    Returns the correlation matrix, the p-value matrix, and a mapping from
    gene name to row/column index.
    """
    values = mrna_expr.values
    corr = np.corrcoef(values)

    n_samples = values.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = corr * np.sqrt((n_samples - 2) / (1 - corr**2 + 1e-10))
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=n_samples - 2))

    gene_index = {gene: i for i, gene in enumerate(mrna_expr.index)}
    return corr, p_values, gene_index


def compute_network_robustness(
    scored_pairs: pd.DataFrame,
    mrna_expr: pd.DataFrame,
    score_thresholds: List[float] = config.SCORE_THRESHOLDS,
    corr_threshold: float = config.COEXPRESSION_CORR_THRESHOLD,
    p_threshold: float = config.COEXPRESSION_P_THRESHOLD,
) -> Dict[float, Dict[str, int]]:
    """Evaluate network size across a range of interaction-score thresholds.

    For each threshold, counts:
      - miRNA-mRNA edges: predicted pairs scoring at or above the threshold.
      - mRNA-mRNA edges: co-expression edges (|r| threshold and significance)
        among the mRNA nodes reachable at that threshold, restricted to gene
        pairs that share a regulating miRNA (a ceRNA "shared regulator"
        relationship).
      - total nodes / edges in the combined network.
    """
    corr, p_values, gene_index = compute_coexpression_matrix(mrna_expr)

    results: Dict[float, Dict[str, int]] = {}
    for threshold in score_thresholds:
        edges = scored_pairs[scored_pairs["score"] >= threshold]
        mirna_mrna_edges = len(edges)

        mrna_edges = 0
        for _, group in edges.groupby("miRNA_ID"):
            targets = group["gene_name"].unique()
            for i in range(len(targets)):
                for j in range(i + 1, len(targets)):
                    idx1, idx2 = gene_index[targets[i]], gene_index[targets[j]]
                    r, p = corr[idx1, idx2], p_values[idx1, idx2]
                    if r > corr_threshold and p < p_threshold:
                        mrna_edges += 1

        total_nodes = len(set(edges["miRNA_ID"]) | set(edges["gene_name"]))
        results[threshold] = {
            "mirna_nodes": edges["miRNA_ID"].nunique(),
            "mrna_nodes": edges["gene_name"].nunique(),
            "mirna_mrna_edges": mirna_mrna_edges,
            "mrna_mrna_edges": mrna_edges,
            "total_edges": mirna_mrna_edges + mrna_edges,
            "total_nodes": total_nodes,
        }
        logger.info("Threshold %.2f: %s", threshold, results[threshold])

    return results


def build_bipartite_network(top_pairs: pd.DataFrame) -> nx.Graph:
    """Build a bipartite miRNA-mRNA graph from a table of (miRNA_ID, gene_name, score) rows."""
    graph = nx.Graph()
    mirnas = top_pairs["miRNA_ID"].unique()
    genes = top_pairs["gene_name"].unique()
    graph.add_nodes_from(mirnas, bipartite=0, node_type="miRNA")
    graph.add_nodes_from(genes, bipartite=1, node_type="mRNA")
    for _, row in top_pairs.iterrows():
        graph.add_edge(row["miRNA_ID"], row["gene_name"], weight=row["score"])
    return graph


def compute_graph_degrees(graph: nx.Graph, node_type: str | None = None) -> pd.DataFrame:
    """Return a DataFrame of node degrees, optionally filtered to one node type."""
    records = [
        {"gene_name": node, "degree": degree, "type": data.get("node_type")}
        for node, degree, data in ((n, graph.degree(n), graph.nodes[n]) for n in graph.nodes)
    ]
    df = pd.DataFrame(records)
    if node_type is not None:
        df = df[df["type"] == node_type]
    return df


def identify_top_hub_genes(
    degree_df: pd.DataFrame,
    top_n: int = config.HUB_TOP_N,
    degree_column: str = "degree",
    name_column: str = "gene_name",
) -> pd.DataFrame:
    """Return the top-N highest-degree genes as candidate regulatory hubs.

    Hub genes must always be derived this way -- by ranking the degree
    values actually computed from the constructed network -- rather than
    supplied as a fixed, hand-picked gene list. This is the single place in
    the pipeline where "hub" status is assigned.
    """
    return degree_df.sort_values(degree_column, ascending=False).head(top_n).reset_index(drop=True)


def plot_bipartite_network(graph: nx.Graph, save_path: Path, seed: int = config.RANDOM_STATE) -> None:
    """Draw a bipartite miRNA-mRNA network using a spring layout."""
    import matplotlib.pyplot as plt

    mirnas = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "miRNA"]
    genes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "mRNA"]
    pos = nx.spring_layout(graph, seed=seed, k=0.5)

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(graph, pos, nodelist=mirnas, node_color="#67001f", node_size=400, label="miRNA", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=genes, node_color="#053061", node_size=400, label="mRNA", ax=ax)

    edges = graph.edges(data=True)
    edge_widths = [abs(d["weight"]) * 5 for (_, _, d) in edges]
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edge_widths, edge_color="black", alpha=0.6, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black", ax=ax)

    ax.legend(loc="lower left", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def find_missing_reference_pairs(
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    mirna_col: str = "miRNA_ID",
    gene_col: str = "gene_name",
) -> pd.DataFrame:
    """Identify (miRNA, gene) pairs present in a reference list but absent from the scored table.

    Both inputs are matched case-insensitively on miRNA identifier
    (lower-cased) and gene symbol (upper-cased). Useful as a sanity check
    that all expected reference interactions were scored by the pipeline.
    """
    ref = reference.copy()
    sc = scored.copy()
    ref[mirna_col] = ref[mirna_col].str.lower()
    ref[gene_col] = ref[gene_col].str.upper()
    sc[mirna_col] = sc[mirna_col].str.lower()
    sc[gene_col] = sc[gene_col].str.upper()

    merged = ref.merge(sc[[mirna_col, gene_col]], on=[mirna_col, gene_col], how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def run_pipeline(group: str, compare_with: str | None) -> None:
    results_dir = config.RESULTS_DIR / group
    figures_dir = config.FIGURES_DIR / group

    scored_pairs = deduplicate_pair_scores(load_group_predictions(group))
    scored_pairs.to_csv(results_dir / "scored_pairs.csv", index=False)

    mrna_path = config.GROUP_A_MRNA_FILE if group == config.GROUP_A_NAME else config.GROUP_B_MRNA_FILE
    mrna_expr = load_mrna_expression(mrna_path)

    robustness = compute_network_robustness(scored_pairs, mrna_expr)
    pd.DataFrame(robustness).T.to_csv(results_dir / "network_robustness.csv")

    # Small, illustrative network (top-scoring pairs) for the bipartite plot
    # and interaction heatmap, where legibility requires a modest node count.
    top_pairs = scored_pairs.sort_values("score", ascending=False).head(config.TOP_N_NETWORK_PAIRS)
    graph = build_bipartite_network(top_pairs)
    plot_bipartite_network(graph, figures_dir / "network_top_pairs.svg")
    visualization.plot_interaction_heatmap(top_pairs, save_path=figures_dir / "interaction_heatmap.svg")

    # Larger network (top TOP_N_HUB_ANALYSIS_PAIRS pairs) used for degree and
    # hub analysis, which needs a wider pool than the small illustrative
    # network above to give a stable degree distribution.
    hub_pairs = scored_pairs.sort_values("score", ascending=False).head(config.TOP_N_HUB_ANALYSIS_PAIRS)
    hub_graph = build_bipartite_network(hub_pairs)
    degree_df = compute_graph_degrees(hub_graph, node_type="mRNA")
    degree_df.to_csv(results_dir / "gene_degree.csv", index=False)

    # Hub genes are always identified automatically from the computed degree
    # table -- never a hard-coded gene list.
    top_hubs = identify_top_hub_genes(degree_df)
    top_hubs.to_csv(results_dir / "top_hub_genes.csv", index=False)
    visualization.plot_selected_feature_degrees(
        degree_df, top_hubs["gene_name"].tolist(), save_path=figures_dir / "top_hub_genes.svg"
    )

    # Optional, purely supplementary highlight of user-chosen genes.
    genes_of_interest = (
        config.GROUP_A_GENES_OF_INTEREST if group == config.GROUP_A_NAME else config.GROUP_B_GENES_OF_INTEREST
    )
    if genes_of_interest:
        visualization.plot_selected_feature_degrees(
            degree_df, genes_of_interest, save_path=figures_dir / "selected_gene_degree.svg"
        )

    if compare_with:
        other_pairs = deduplicate_pair_scores(load_group_predictions(compare_with))
        other_mrna_path = config.GROUP_A_MRNA_FILE if compare_with == config.GROUP_A_NAME else config.GROUP_B_MRNA_FILE
        other_mrna_expr = load_mrna_expression(other_mrna_path)
        other_robustness = compute_network_robustness(other_pairs, other_mrna_expr)

        thresholds = config.SCORE_THRESHOLDS
        counts_by_group = {
            group: {
                "miRNA": [robustness[t]["mirna_nodes"] for t in thresholds],
                "mRNA": [robustness[t]["mrna_nodes"] for t in thresholds],
                "Edges": [robustness[t]["total_edges"] for t in thresholds],
            },
            compare_with: {
                "miRNA": [other_robustness[t]["mirna_nodes"] for t in thresholds],
                "mRNA": [other_robustness[t]["mrna_nodes"] for t in thresholds],
                "Edges": [other_robustness[t]["total_edges"] for t in thresholds],
            },
        }
        visualization.plot_threshold_sensitivity(
            thresholds, counts_by_group, save_path=config.FIGURES_DIR / "threshold_sensitivity_comparison.svg"
        )

        other_hub_pairs = other_pairs.sort_values("score", ascending=False).head(config.TOP_N_HUB_ANALYSIS_PAIRS)
        other_degree_df = compute_graph_degrees(build_bipartite_network(other_hub_pairs), node_type="mRNA")
        visualization.plot_degree_distribution(
            {group: degree_df["degree"], compare_with: other_degree_df["degree"]},
            save_path=config.FIGURES_DIR / "degree_distribution_comparison.svg",
        )

    logger.info("Network analysis outputs saved to %s and %s", results_dir, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=[config.GROUP_A_NAME, config.GROUP_B_NAME], required=True)
    parser.add_argument(
        "--compare-with",
        choices=[config.GROUP_A_NAME, config.GROUP_B_NAME],
        default=None,
        help="Optional second group to compare network robustness and degree distribution against.",
    )
    args = parser.parse_args()
    run_pipeline(args.group, args.compare_with)


if __name__ == "__main__":
    main()

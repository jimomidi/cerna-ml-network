"""
Load and preprocess miRNA/mRNA expression data and known miRNA-target
interaction pairs, and assemble the feature/target dataset used to train
the interaction-strength regression model.

Expected input formats
-----------------------
miRNA expression (long format):
    columns: miRNA_ID, case_id, reads_per_million_miRNA_mapped

mRNA expression (long format):
    columns: gene_id, gene_name, case_id, tpm_unstranded

Known interactions (e.g. a miRTarBase-style export):
    columns: "Species (miRNA)", "miRNA", "Target Gene"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config

logger = logging.getLogger(__name__)


def load_mirna_expression(
    path: Path,
    expression_threshold: float = config.EXPRESSION_THRESHOLD,
    lowercase_index: bool = config.NORMALIZE_INTERACTION_IDS,
) -> pd.DataFrame:
    """Load a long-format miRNA expression file into a feature-by-sample matrix.

    Rows are miRNA identifiers, columns are sample identifiers. Low-expression
    miRNAs (mean value across samples below ``expression_threshold``) are
    removed. When ``lowercase_index`` is True (the default, kept in sync with
    ``config.NORMALIZE_INTERACTION_IDS``), identifiers are lower-cased so
    they match the normalized identifiers produced by
    ``load_known_interactions``.
    """
    df = pd.read_csv(path)
    expr = df.pivot(
        index="miRNA_ID", columns="case_id", values="reads_per_million_miRNA_mapped"
    )
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr[expr.mean(axis=1) >= expression_threshold]
    if lowercase_index:
        expr.index = expr.index.str.lower()
    return expr


def load_mrna_expression(
    path: Path,
    expression_threshold: float = config.EXPRESSION_THRESHOLD,
    lowercase_index: bool = config.NORMALIZE_INTERACTION_IDS,
) -> pd.DataFrame:
    """Load a long-format mRNA expression file into a feature-by-sample matrix.

    Rows are gene symbols, columns are sample identifiers. Non-gene summary
    rows produced by standard RNA-seq quantification pipelines (identified by
    a "N_" prefixed gene_id, e.g. ambiguous/unmapped read counters) are
    dropped, gene-id version suffixes are stripped, and duplicate gene
    symbols are aggregated by mean. Low-expression genes (mean value across
    samples below ``expression_threshold``) are removed. When
    ``lowercase_index`` is True (the default, kept in sync with
    ``config.NORMALIZE_INTERACTION_IDS``), gene symbols are lower-cased so
    they match the normalized identifiers produced by
    ``load_known_interactions``.
    """
    df = pd.read_csv(path)
    df = df[~df["gene_id"].str.startswith("N_")].copy()
    df["gene_id"] = df["gene_id"].str.split(".").str[0]

    expr = df.pivot_table(
        index="gene_name", columns="case_id", values="tpm_unstranded", aggfunc="mean"
    )
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr[expr.mean(axis=1) >= expression_threshold]
    if lowercase_index:
        expr.index = expr.index.str.lower()
    return expr


def align_samples(
    mirna_expr: pd.DataFrame, mrna_expr: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both expression matrices to their shared set of samples."""
    common_samples = sorted(set(mirna_expr.columns) & set(mrna_expr.columns))
    if not common_samples:
        raise ValueError("No overlapping samples found between miRNA and mRNA data.")
    return (
        mirna_expr[common_samples].dropna(),
        mrna_expr[common_samples].dropna(),
    )


def load_known_interactions(
    path: Path,
    mirna_index: pd.Index,
    gene_index: pd.Index,
    species: str | None = config.INTERACTION_SPECIES,
    normalize_ids: bool = config.NORMALIZE_INTERACTION_IDS,
) -> pd.DataFrame:
    """Load and filter a known miRNA-target interaction table.

    Restricts the interactions to those whose species tag matches
    ``species`` (when provided) and whose miRNA/gene identifiers are present
    in the supplied expression matrices. When ``normalize_ids`` is True,
    identifiers are lower-cased and a trailing "-3p"/"-5p" arm suffix is
    stripped from the miRNA identifier before matching, which increases the
    number of matched pairs.

    Returns a DataFrame with columns ["miRNA_ID", "gene_name"].
    """
    df = pd.read_csv(path)

    if species is not None:
        df = df[df["Species (miRNA)"] == species].copy()

    if normalize_ids:
        df["miRNA_ID"] = df["miRNA"].str.replace(r"-(3p|5p)$", "", regex=True).str.lower()
        df["gene_name"] = df["Target Gene"].str.lower()
    else:
        df = df.rename(columns={"miRNA": "miRNA_ID", "Target Gene": "gene_name"})

    df = df[["miRNA_ID", "gene_name"]]
    df = df[df["miRNA_ID"].isin(mirna_index)]
    df = df[df["gene_name"].isin(gene_index)]
    return df.drop_duplicates().reset_index(drop=True)


def build_feature_target_dataset(
    mirna_expr: pd.DataFrame,
    mrna_expr: pd.DataFrame,
    pairs: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Build the regression dataset from expression data and candidate pairs.

    For each miRNA-gene pair, the feature vector is the concatenation of the
    miRNA and mRNA expression profiles across samples, and the regression
    target is the Pearson correlation coefficient between the two profiles.
    Pairs where either profile has zero variance are skipped.

    Returns
    -------
    X : np.ndarray of shape (n_pairs, 2 * n_samples)
    y : np.ndarray of shape (n_pairs,)
    pair_list : list of (miRNA_ID, gene_name) tuples, aligned with X and y.
    """
    features, targets, pair_list = [], [], []

    for _, row in pairs.iterrows():
        mirna, gene = row["miRNA_ID"], row["gene_name"]

        mirna_vec = mirna_expr.loc[mirna].values.astype(float)
        gene_vec = mrna_expr.loc[gene].values.astype(float)

        if np.std(mirna_vec) == 0 or np.std(gene_vec) == 0:
            continue

        features.append(np.concatenate([mirna_vec, gene_vec]))
        targets.append(pearsonr(mirna_vec, gene_vec)[0])
        pair_list.append((mirna, gene))

    return np.array(features), np.array(targets), pair_list


def load_group_dataset(
    mirna_path: Path,
    mrna_path: Path,
    interactions_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline for a single sample group.

    Returns X, y, pair_list, and the two aligned expression matrices (which
    are needed downstream for network construction and interpretation).
    """
    logger.info("Loading miRNA expression from %s", mirna_path)
    mirna_expr = load_mirna_expression(mirna_path)

    logger.info("Loading mRNA expression from %s", mrna_path)
    mrna_expr = load_mrna_expression(mrna_path)

    mirna_expr, mrna_expr = align_samples(mirna_expr, mrna_expr)

    logger.info("Loading known interactions from %s", interactions_path)
    pairs = load_known_interactions(interactions_path, mirna_expr.index, mrna_expr.index)
    logger.info("Matched %d candidate interaction pairs", len(pairs))

    X, y, pair_list = build_feature_target_dataset(mirna_expr, mrna_expr, pairs)
    logger.info("Built feature matrix with shape %s", X.shape)

    return X, y, pair_list, mirna_expr, mrna_expr

"""
Central configuration for the ceRNA ML network pipeline.

All paths, filenames, and tunable parameters used across the pipeline are
defined here so that a full analysis can be reproduced or adapted to a new
dataset by editing a single file.

All paths are relative to the project root and portable across machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------
# Directory layout
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for directory in (RESULTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Input files
# --------------------------------------------------------------------------
# The pipeline compares two sample groups (e.g. "case" vs. "control").
# Expression files are expected in long format, matching the standard
# GDC/TCGA quantification output:
#
#   miRNA expression : miRNA_ID, case_id, reads_per_million_miRNA_mapped
#   mRNA expression  : gene_id, gene_name, case_id, tpm_unstranded
#
# The known-interactions file is expected to follow the miRTarBase export
# schema: "Species (miRNA)", "miRNA", "Target Gene".

GROUP_A_NAME = "case"
GROUP_B_NAME = "control"

GROUP_A_MIRNA_FILE = DATA_DIR / "case_mirna_expression.csv"
GROUP_A_MRNA_FILE = DATA_DIR / "case_mrna_expression.csv"

GROUP_B_MIRNA_FILE = DATA_DIR / "control_mirna_expression.csv"
GROUP_B_MRNA_FILE = DATA_DIR / "control_mrna_expression.csv"

KNOWN_INTERACTIONS_FILE = DATA_DIR / "known_interactions.csv"

# Optional file used only by the validation utility in network_analysis.py.
REFERENCE_PAIRS_FILE = DATA_DIR / "reference_pairs.csv"

# --------------------------------------------------------------------------
# Preprocessing parameters
# --------------------------------------------------------------------------

# Minimum mean expression (across samples) required to keep a feature.
EXPRESSION_THRESHOLD = 0.01

# Restrict known interactions to this species tag in the "Species (miRNA)"
# column (set to None to disable species filtering).
INTERACTION_SPECIES = "hsa"

# If True, strip a trailing "-3p"/"-5p" arm suffix and lower-case miRNA and
# gene identifiers before matching them against the expression matrices.
# This produces a more permissive (case- and arm-insensitive) match.
NORMALIZE_INTERACTION_IDS = True

# --------------------------------------------------------------------------
# Model training parameters
# --------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5

# Hyperparameters for the production model (RandomForestRegressor).
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# --------------------------------------------------------------------------
# Network construction / robustness parameters
# --------------------------------------------------------------------------

# Predicted-interaction score thresholds used for network robustness /
# sensitivity analysis.
SCORE_THRESHOLDS = [0.7, 0.8, 0.9]

# Threshold used to call an mRNA-mRNA co-expression edge (Pearson r and
# two-sided p-value).
COEXPRESSION_CORR_THRESHOLD = 0.2
COEXPRESSION_P_THRESHOLD = 0.05

# Number of top-scoring pairs to visualize as a bipartite network / heatmap
# (Sankey-style / small illustrative networks).
TOP_N_NETWORK_PAIRS = 50

# Number of top-scoring pairs used to build the larger network from which
# node degree and hub genes are computed. A larger pool than
# TOP_N_NETWORK_PAIRS gives a more stable degree distribution.
TOP_N_HUB_ANALYSIS_PAIRS = 10_000

# Number of top-degree genes to report/plot as candidate regulatory hubs.
# Hub genes are always identified automatically from the computed degree
# table (see network_analysis.identify_top_hub_genes) -- never hard-coded.
HUB_TOP_N = 10

# --------------------------------------------------------------------------
# Model interpretation (SHAP) parameters
# --------------------------------------------------------------------------

# SHAP analysis is restricted to high-confidence pairs (predicted score
# above this threshold) to keep the interpretation focused on biologically
# meaningful interactions, consistent with the network confidence threshold.
SHAP_SCORE_THRESHOLD = 0.7

SHAP_BEESWARM_MAX_DISPLAY = 20
SHAP_SUMMARY_MAX_DISPLAY = 10
FEATURE_IMPORTANCE_TOP_N = 10

# Feature name to use for the SHAP dependence plot. Set to a feature that
# actually exists in your dataset (e.g. a specific miRNA identifier), or
# leave as None to skip the dependence plot.
SHAP_DEPENDENCE_FEATURE: Optional[str] = None

# --------------------------------------------------------------------------
# Optional manual highlight genes (supplementary to automatic hub detection)
# --------------------------------------------------------------------------
# Hub/high-degree genes are always identified automatically from the
# computed network (see network_analysis.identify_top_hub_genes). These
# lists are an entirely optional, additional set of gene symbols you may
# want to highlight for domain reasons unrelated to degree ranking. Left
# empty by default -- no genes are highlighted unless you populate this.

GROUP_A_GENES_OF_INTEREST: List[str] = []
GROUP_B_GENES_OF_INTEREST: List[str] = []

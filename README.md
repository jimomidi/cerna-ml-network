# ceRNA ML Network

Machine learning pipeline for predicting miRNA-mRNA regulatory interaction
strength from paired expression data, and for constructing, interpreting,
and comparing the resulting competing endogenous RNA (ceRNA) networks
across two sample groups (e.g. a case group and a control group).

## Overview

Sequence-based miRNA target prediction tools generate large numbers of
putative interactions with high false-positive rates and generally ignore
condition-specific expression context. This project instead learns a
regression model that predicts the expression correlation between a miRNA
and a candidate target gene directly from their expression profiles, using
a set of literature-curated interactions as training pairs. The trained
model is then used to:

1. Score every candidate miRNA-gene pair for both sample groups.
2. Interpret the model with SHAP and feature-importance analysis.
3. Build a bipartite miRNA-mRNA network from high-confidence pairs.
4. Evaluate how network size (nodes/edges) changes across score thresholds.
5. Compare network structure and node-degree distributions between groups.

## Workflow

```
data_processing.py        Shared preprocessing utilities (imported by the scripts below)
        |
        v
train_models.py            1. Build features/targets, benchmark regression models,
                               train the production model, save artifacts + diagnostics
        |
        v
model_interpretation.py    2. SHAP-based interpretation of the trained model
        |
        v
network_analysis.py        3. Score all pairs, build the network, threshold
                               robustness analysis, network/degree visualizations
```

`visualization.py` holds shared, reusable plotting functions used by all
three pipeline stages, and `config.py` centralizes every path and tunable
parameter.

## Method summary

- **Feature construction**: for each candidate (miRNA, gene) pair, the
  feature vector is the concatenation of the miRNA's and the gene's
  expression profiles across matched samples; the regression target is the
  Pearson correlation coefficient between the two profiles.
- **Model**: a `RandomForestRegressor` (100 trees) is the production model,
  selected after benchmarking against linear regression, gradient boosting,
  LightGBM, and XGBoost on a held-out test split (80/20, fixed random seed).
- **Interpretation**: SHAP (beeswarm, violin, heatmap, dependence plots) is
  computed on the subset of pairs the model scores above
  `SHAP_SCORE_THRESHOLD` (default 0.7), keeping the explanation focused on
  the same high-confidence interactions used for network construction.
  Built-in RF feature importances (impurity-based) are computed on the full
  training set.
- **Network construction**: pairs scoring above a threshold become edges in
  a bipartite miRNA-mRNA graph. Robustness is assessed by sweeping the
  score threshold and, at each threshold, additionally counting
  significant mRNA-mRNA co-expression edges among genes that share a
  regulating miRNA (a ceRNA "shared regulator" relationship).
- **Hub identification**: candidate regulatory hub genes are always
  identified automatically, by ranking node degree in the constructed
  network (`network_analysis.identify_top_hub_genes`) -- there is no
  hard-coded gene list anywhere in the pipeline. Degree/hub analysis is run
  on a larger pool of top-scoring pairs (`TOP_N_HUB_ANALYSIS_PAIRS`) than
  the small network used for illustrative bipartite/heatmap plots
  (`TOP_N_NETWORK_PAIRS`), since hub ranking needs a wider node pool to be
  stable.
- **Group comparison**: the same pipeline is run independently for each
  sample group, and `network_analysis.py --compare-with` overlays the two
  groups' threshold-sensitivity and degree-distribution results.

All thresholds, filtering criteria, the train/test split ratio, and the
random seed are defined once in `config.py`.

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies.

```bash
pip install -r requirements.txt
```

## Expected input data

Place your input files under a local `data/` directory (not tracked by
git). Two versions of each expression file are expected, one per sample
group:

| File | Format | Required columns |
|---|---|---|
| `data/case_mirna_expression.csv` | long format | `miRNA_ID`, `case_id`, `reads_per_million_miRNA_mapped` |
| `data/case_mrna_expression.csv` | long format | `gene_id`, `gene_name`, `case_id`, `tpm_unstranded` |
| `data/control_mirna_expression.csv` | long format | same as above |
| `data/control_mrna_expression.csv` | long format | same as above |
| `data/known_interactions.csv` | curated interaction list | `Species (miRNA)`, `miRNA`, `Target Gene` |

- `case_id` is any sample/subject identifier; it only needs to be
  consistent between the miRNA and mRNA files within a group.
- The expression file schema follows a standard GDC/TCGA-style RNA-seq and
  miRNA-seq quantification output. Any dataset exported in this format can
  be used.
- `known_interactions.csv` follows the export schema of public miRNA-target
  databases such as miRTarBase; only rows matching `INTERACTION_SPECIES` in
  `config.py` (default `"hsa"`) are used.
- (Optional) `data/reference_pairs.csv` with columns `miRNA_ID`, `gene_name`
  can be used with `network_analysis.find_missing_reference_pairs()` to
  sanity-check that all expected reference interactions were scored.

Update the paths, group names, and thresholds in `config.py` if your data
is organized differently.

This pipeline does not perform any batch-effect correction or cross-cohort
normalization itself; it expects `data/*_expression.csv` to already contain
normalized, analysis-ready expression values. If your two groups come from
different sources/platforms/cohorts, apply appropriate normalization and
batch-effect assessment (or correction) during your own data preparation
before running this pipeline.

## Usage

Run the pipeline once per sample group:

```bash
python train_models.py --group case
python train_models.py --group control

python model_interpretation.py --group case
python model_interpretation.py --group control

python network_analysis.py --group case --compare-with control
```

Each script can also be run for a single group without `--compare-with` to
produce group-specific outputs only.

## Outputs

All outputs are written under `results/` (created automatically, not
tracked by git):

```
results/
├── <group>/
│   ├── model.pkl                 Trained RandomForestRegressor
│   ├── model_comparison.csv      Benchmarked model metrics (MSE, MAE, R2)
│   ├── metrics.txt               Held-out and cross-validated performance
│   ├── X_all.npy, y_all.npy      Full feature matrix / targets
│   ├── y_test.npy, y_pred.npy    Held-out predictions
│   ├── pairs.pkl                 (miRNA, gene) pairs aligned with X_all/y_all
│   ├── feature_names.csv         Filtered miRNA/gene identifiers (reference only, see below)
│   ├── scored_pairs.csv          Deduplicated predicted scores for every pair
│   ├── network_robustness.csv    Node/edge counts per score threshold
│   ├── gene_degree.csv           mRNA node degree in the hub-analysis network
│   └── top_hub_genes.csv         Top-N genes by degree (automatically identified)
└── figures/
    ├── <group>/                  Per-group diagnostic, SHAP, and hub figures (.svg)
    ├── threshold_sensitivity_comparison.svg   (when --compare-with is used)
    └── degree_distribution_comparison.svg     (when --compare-with is used)
```

## Reproducibility

- Random seed (`RANDOM_STATE = 42`) is fixed for the train/test split, the
  RandomForest model, and every other stochastic step, and is defined once
  in `config.py`.
- All filtering thresholds (expression threshold, correlation/p-value
  thresholds, score thresholds) are centralized in `config.py`.
- All paths are relative to the project root, so the repository can be
  cloned and run on any machine without editing hardcoded paths.

## Customization

- **Number of automatically identified hub genes**: set `HUB_TOP_N` in
  `config.py` (default 10).
- **Optional supplementary gene highlight**: set
  `GROUP_A_GENES_OF_INTEREST` / `GROUP_B_GENES_OF_INTEREST` in `config.py`
  to additionally plot degree for specific gene symbols of your own choosing
  (empty by default; this is independent of, and never a substitute for,
  the automatic hub identification above).
- **SHAP dependence plot**: set `SHAP_DEPENDENCE_FEATURE` in `config.py` to
  a feature present in your dataset (`None` skips this plot).
- **SHAP confidence filter**: set `SHAP_SCORE_THRESHOLD` in `config.py` to
  change which predicted pairs are included in the SHAP analysis (default
  0.7, matching the network confidence threshold).
- **Score/correlation thresholds, top-N network sizes, model
  hyperparameters**: all defined in `config.py`.

## Known limitations

- The model's input features are the concatenated per-sample expression
  values of a miRNA-gene pair (length `2 x n_samples`), not a fixed
  per-gene feature space. As a result, the reference identifier list saved
  to `feature_names.csv` (every miRNA/gene that passed the expression
  filter) will only line up one-to-one with the model's actual input
  columns when that identifier count happens to equal `2 x n_samples`. When
  it does not, `model_interpretation.py` detects the mismatch, logs a
  warning, and falls back to generic column labels (`feature_0`,
  `feature_1`, ...) for SHAP and feature-importance plots so that labels
  are never silently misassigned. If you want SHAP output labeled by
  sample, rename columns accordingly before interpreting results.
- The mRNA-mRNA co-expression significance test uses a normal-theory
  t-approximation for the Pearson correlation p-value; it is a
  large-sample approximation and does not correct for multiple testing.

## License

Released under the MIT License. See `LICENSE`.

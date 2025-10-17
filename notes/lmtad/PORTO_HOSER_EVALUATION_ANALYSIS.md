# Evaluation Analysis: Porto HOSER

**Generated:** 2025-10-14 17:50:53  
**Run ID:** `run_20251010_212829`  
**Evaluation Directory:** `porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval`

---

## 1. Overview

This document presents a comprehensive analysis of the LMTAD model evaluation on the Porto HOSER dataset. The evaluation assesses the model's ability to detect trajectory anomalies using log perplexity as the primary metric.

### Model Configuration

- Model configuration extracted from checkpoint

---

## 2. Dataset Statistics

### Trajectory Counts

| Category | Count | Percentage |
|----------|-------|------------|
| **Total Trajectories** | 639,844 | 100.0% |
| Non-outlier | 597,993 | 93.46% |
| Route Switch Outliers | 20,927 | 3.27% |
| Detour Outliers | 20,924 | 3.27% |

### Sequence Length Statistics

| Trajectory Type | Min | Max | Mean | Std Dev |
|----------------|-----|-----|------|---------|
| **Non-outlier** | 21 | 200 | 44.9 | 18.6 |
| **Route Switch** | 21 | 167 | 44.9 | 18.7 |
| **Detour** | 21 | 319 | 83.4 | 36.5 |

### Log Perplexity Statistics

| Trajectory Type | Mean | Std Dev | Median |
|----------------|------|---------|--------|
| **Non-outlier** | 0.3822 | 0.1249 | 0.3649 |
| **Route Switch** | 7.0265 | 1.6068 | 7.0401 |
| **Detour** | 8.4132 | 1.2098 | 8.5035 |

---

## 3. Performance Metrics Summary

### Outlier Detection Performance
*Metrics calculated excluding route switch outliers (detour vs. non-outlier classification)*

| Configuration | Ratio | Level | Prob | Threshold | Accuracy | Precision | Recall | F1 Score | Avg Precision | PR-AUC |
|--------------|-------|-------|------|-----------|----------|-----------|--------|----------|---------------|--------|
| ckpt_best | 0.05 | 3 | 0.3 | 0.7571 | 0.9934 | 0.8366 | 0.9999 | 0.9110 | 0.8366 | 0.9999 |
| ckpt_best | 0.05 | 5 | 0.1 | 0.7571 | 0.9924 | 0.8325 | 0.9705 | 0.8963 | 0.8090 | 0.9764 |
| ckpt_best | 0.05 | 3 | 0.1 | 0.7571 | 0.9923 | 0.8322 | 0.9680 | 0.8950 | 0.8066 | 0.9746 |

---

## 4. Visualizations

### 4.1 Performance Metrics

#### Metrics Heatmap
Comprehensive view of precision, recall, F1, and PR-AUC across configurations:

![Metrics Heatmap](../../assets/plots/lmtad/metrics.png)

#### ROC Curves
Receiver Operating Characteristic curves showing true positive rate vs. false positive rate:

![ROC Curves](../../assets/plots/lmtad/roc_curves.png)

#### Precision-Recall Curves
Precision-Recall trade-offs across different threshold values:

![Precision-Recall Curves](../../assets/plots/lmtad/pr_curves.png)

---

### 4.2 Distribution Analysis

#### Log Perplexity Distributions
KDE plots showing how perplexity values distribute across trajectory types:

![Perplexity Distributions](../../assets/plots/lmtad/distribution_histograms.png)

**Key Observations:**
- Green dotted line: Non-outlier trajectories
- Red/Orange filled: Route switch outliers  
- Blue filled: Detour outliers

#### Perplexity Box Plots
Statistical summary of perplexity distributions by trajectory type:

![Perplexity Box Plots](../../assets/plots/lmtad/distribution_boxplots.png)

#### Sequence Length Distributions
KDE plots showing trajectory length distributions:

![Sequence Length Distributions](../../assets/plots/lmtad/sequence_length_distributions.png)

**Key Observations:**
- Route switch outliers preserve similar lengths to non-outliers
- Detour outliers tend to have longer trajectories

---

### 4.3 Scatter Plot Analysis

#### All Outliers Comparison
Log perplexity vs. sequence length for all trajectory types:

![All Outliers Scatter](../../assets/plots/lmtad/scatter_all_outliers.png)

#### Route Switch Outliers Focus
Detailed view of route switch outliers vs. non-outliers:

![Route Switch Scatter](../../assets/plots/lmtad/scatter_route_switch.png)

#### Detour Outliers Focus
Detailed view of detour outliers vs. non-outliers:

![Detour Scatter](../../assets/plots/lmtad/scatter_detour.png)

---

## 5. Evaluation Configurations

The model was evaluated using the following outlier injection configurations:


### Configuration: ckpt_best
- **Outlier Ratio:** 0.05 (proportion of outliers in dataset)
- **Outlier Level:** 3 (magnitude/severity of anomaly)
- **Outlier Probability:** 0.3 (probability parameter for generation)
- **Detection Threshold:** 0.7571 (optimal threshold for classification)

### Configuration: ckpt_best
- **Outlier Ratio:** 0.05 (proportion of outliers in dataset)
- **Outlier Level:** 5 (magnitude/severity of anomaly)
- **Outlier Probability:** 0.1 (probability parameter for generation)
- **Detection Threshold:** 0.7571 (optimal threshold for classification)

### Configuration: ckpt_best
- **Outlier Ratio:** 0.05 (proportion of outliers in dataset)
- **Outlier Level:** 3 (magnitude/severity of anomaly)
- **Outlier Probability:** 0.1 (probability parameter for generation)
- **Detection Threshold:** 0.7571 (optimal threshold for classification)

---

## 6. Key Findings

### Model Performance
<!-- TODO: Add interpretation of overall model performance -->

### Distribution Insights
<!-- TODO: Analyze the distribution patterns observed -->

### Outlier Type Comparison
<!-- TODO: Compare how the model performs on different outlier types -->

### Sequence Length Impact
<!-- TODO: Discuss relationship between sequence length and detection -->

---

## 7. Notes & Interpretations

### Observations
<!-- TODO: Add detailed observations from the evaluation -->

### Strengths
<!-- TODO: Identify what the model does well -->

### Limitations
<!-- TODO: Note any limitations or failure modes -->

### Future Work
<!-- TODO: Suggestions for improvements or further analysis -->

---

## Appendix

### Files in This Directory

**Visualizations:**
- `distribution_boxplots.png`
- `distribution_histograms.png`
- `metrics.png`
- `pr_curves.png`
- `roc_curves.png`
- `scatter_all_outliers.png`
- `scatter_detour.png`
- `scatter_route_switch.png`
- `sequence_length_distributions.png`

**Data Files:**
- `ckpt_best_outliers_config_ratio_0.05_level_3_prob_0.1.tsv`
- `ckpt_best_outliers_config_ratio_0.05_level_3_prob_0.3.tsv`
- `ckpt_best_outliers_config_ratio_0.05_level_5_prob_0.1.tsv`

### Analysis Generation
- **Generated by:** `generate_eval_analysis.py`
- **Date:** 2025-10-14 17:50:53

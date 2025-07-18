---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Data Processing & Model Selection | Mateusz Kędzia'
footer: 'Matts MSc AI Thesis | July 2025'
style: |
  section {
    font-size: 22px;
  }
  h1 {
    font-size: 34px;
  }
  h2 {
    font-size: 30px;
  }
  h3 {
    font-size: 26px;
  }
  ul, ol {
    font-size: 20px;
  }
  table {
    font-size: 18px;
  }
  blockquote {
    font-size: 20px;
  }
---

<!-- _class: lead -->
# Data Processing & Model Selection Update
## Privacy-Preserving Synthetic Trajectory Generation
**Mateusz Kędzia** | MSc Artificial Intelligence | Vrije Universiteit Amsterdam
*Progress Update - Week 29 - July 2025*

---

# Overview

**Project Status Update:**
- **Section 1:** Data Processing - Large-scale GPS dataset cleaning completed
- **Section 2:** Model Selection - Evidence-based choice of trajectory generation approach  
- **Section 3:** Chosen Model Implementation - Technical details and performance validation

**Key Outcomes:**
1. Clean, analysis-ready dataset (1.16B GPS records)
2. Optimal model selection with clear justification
3. Implementation roadmap for synthetic trajectory generation

---

<!-- _class: lead -->
# **SECTION 1**
## Data Processing & Dataset Preparation

---

# Data Processing Challenge & Solution

**INPUT:** 7 Parquet files, 1.16 billion raw GPS records

**PROBLEM:** 
- Schema inconsistencies across files
- Invalid timestamps and GPS coordinates  
- Performance bottlenecks preventing analysis

**PROCESS:** DuckDB-based cleaning pipeline
- File-by-file standardization with type coercion
- Safe conversion using `TRY_CAST` (invalid → `NULL`)
- Optimized Parquet output format

**OUTPUT:** Clean, schema-consistent dataset ready for model training

---

# Dataset Characteristics

| **Data Source** | Beijing taxi GPS data (25.11.2019 - 01.12.2019) |
|-----------------|---------------------------------------------------|
| **Raw Data Volume** | 7 Parquet files, 1.16 billion GPS records |
| **Daily Data Size** | ~16GB per day of raw GPS data |
| **Temporal Coverage** | Late Nov/early Dec 2019 (concentrated 7-day period) |
| **Geographic Center** | Beijing metropolitan area (115.6° E, 39.8° N) |
| **Processing Engine** | DuckDB with 12GB memory, 8 threads optimization |
| **Vehicle Behavior** | High idle time (median speed 0.0 km/h) |
| **Fleet Distribution** | ZHTC and JYJ companies dominate (>60% of records) |
| **Data Quality Issues** | 275,000+ invalid GPS coordinates filtered out |
| **Processing Result** | Schema-consistent, analysis-ready dataset |

---

<!-- _class: lead -->
# **SECTION 2** 
## Model Selection & Comparison

---


# Model Comparison Results

| Aspect | DiffTraj | Diff-RNTraj | **HOSER** |
|--------|----------|-------------|-----------|
| **Input Format** | Manual conversion needed | **GPS trajectories (automatic mapping)** | **Direct GPS support** |
| **Preprocessing** | Not provided (user responsibility) | **Built-in map-matching pipeline** | **Built-in with minimal setup** |
| **Output Format** | GPS coordinates (manual post-processing) | Road segment IDs (requires network mapping) | **Same as input (GPS + metadata)** |
| **Integration Effort** | High (manual preprocessing + post-processing) | Moderate (ID-to-network conversion for analysis) | **Low (seamless GPS workflow)** |
| **Code & Data** | Available on GitHub | **Available on GitHub** | **Available on GitHub** |

**RESULT:** HOSER selected as optimal choice for taxi GPS trajectory generation

---

# Technical Architecture & Research Details

| Aspect | DiffTraj (2023) | Diff-RNTraj (2024) | **HOSER (2025)** |
|--------|-----------------|---------------------|------------------|
| **Publication** | October 2023 (NeurIPS) | September 2024 (TKDE) | **February 2025 (AAAI)** |
| **Architecture** | 1D-CNN + Trajectory UNet | Diffusion + Residual Dilation Layers | **Road Encoder + Multi-Granularity + Navigator** |
| **Innovation** | First diffusion for GPS trajectories | Road network-constrained generation | **Holistic semantic + destination guidance** |
| **Components** | Wide & Deep + conditional embedding | Node2Vec + RDCL denoising | **GATv2 + GCN + Transformer + Attention** |
| **Datasets** | Chengdu, Xi'an, Porto | Porto, Chengdu | **Beijing, Porto, San Francisco** |
| **Data Format** | GPS coordinates | Road segments + ratios | **GPS + network + temporal data** |
| **Privacy Claims** | Synthetic generation = inherent privacy | Synthetic generation = inherent privacy | **Synthetic generation = inherent privacy** |

**ADVANTAGE:** Latest architecture with comprehensive semantic modeling

---

# HOSER Selection Justification

**WHY HOSER WINS:**

**Technical Architecture Advantages:**
- **Latest Innovation (2025):** Holistic semantic + destination guidance approach
- **Advanced Components:** GATv2 + GCN + Transformer + Attention mechanisms
- **Multi-Granularity:** Road-level + Zone-level + Point-level + Trajectory-level encoding
- **Destination-Oriented:** Additive attention for trajectory + destination guidance

**Implementation Advantages:**
- **Minimal Data Transformation:** Direct GPS trajectory ingestion without format conversion
- **Rich Metadata Utilization:** Leverages timestamps, speed, angle for contextual information
- **Output Consistency:** Generated trajectories match input structure seamlessly
- **Built-in Preprocessing:** Integrated map-matching pipeline with minimal setup

**Practical Advantages:** Fastest implementation path with lowest technical risk

---

<!-- _class: lead -->
# **SECTION 3**
## HOSER: Technical Details & Performance

---

# HOSER Architecture Overview

**INPUT:** Origin-destination pairs + starting time + OpenStreetMap road networks

**THREE-COMPONENT FRAMEWORK:**

**1. Road Network Encoder:**
- **Road-Level:** GATv2 for local segment relationships
- **Zone-Level:** GCN for traffic zone connectivity

**2. Multi-Granularity Trajectory Encoder:**  
- **Point-Level:** Spatio-temporal semantics with Fourier encoding
- **Trajectory-Level:** Transformer with relative position encoding

**3. Destination-Oriented Navigator:**
- **Integration:** Additive attention for trajectory + destination guidance
- **Features:** Distance and angle metrics for realistic route selection

**OUTPUT:** High-fidelity synthetic GPS trajectories

---

# HOSER Performance & Advantages (As described in the paper)

**PERFORMANCE vs. DiffTraj:**
- **Global Metrics:** 27% better distance distribution similarity
- **Local Metrics:** 12% improvement in Hausdorff distance accuracy  
- **Downstream Tasks:** Superior utility (88.8% vs. 64.8% Acc@5)
- **Generalization:** Effective across Beijing, Porto, San Francisco

**KEY CAPABILITIES:**
- **Few-Shot Learning:** Works with only 5,000 training trajectories
- **Zero-Shot Transfer:** Generalizes across cities without retraining
- **High Fidelity:** Generated trajectories match real distribution patterns

**PRIVACY BENEFITS:**
- **Synthetic Generation:** Addresses data access restrictions
- **Regulatory Compliance:** Research without sensitive data exposure
- **Utility Preservation:** Maintains patterns needed for anomaly detection

---


# Approach Change: Pipeline Focus

**CONTRARY TO PREVIOUS PLAN:**

**NEW FOCUS:** Construct privacy-oriented pipeline for synthetic labeled data generation

**KEY CHANGE:** No model retraining - use pre-trained models as designed

**PIPELINE OBJECTIVES:**
- **Privacy-Preserving:** Synthetic data generation without real data exposure
- **Labeled Dataset:** Automatic anomaly detection and categorization
- **Evaluation Ready:** Complete dataset for downstream analysis

**WORKFLOW:** Real Data → Pre-trained Models → Synthetic Labeled Data → Evaluation

**BENEFITS:**
- **Faster Implementation:** No complex retraining cycles
- **Privacy Compliant:** Synthetic data addresses access restrictions
- **Research Ready:** Labeled dataset enables anomaly detection research

---

# Synthetic Labeled Dataset: Complete Pipeline

**STEP 1: HOSER Synthetic Trajectory Generation**
- **INPUT:** Clean GPS dataset
- **PROCESS:** HOSER model training and trajectory generation
- **OUTPUT:** Synthetic trajectory dataset

**STEP 2: LM-TAD Anomaly Detection**
- **INPUT:** Synthetic trajectories from HOSER
- **PROCESS:** LM-TAD Perplexity scoring - higher perplexity = more anomalous
- **OUTPUT:** Synthetic trajectory dataset + anomaly flags

**STEP 3: Rule-Based Categorization**
- **INPUT:** LM-TAD flagged anomalous trajectories  
- **PROCESS:** Apply mathematical rules to categorize anomaly types
- **OUTPUT:** Anomalous trajectory labeles

**WORKFLOW:** Clean Data → HOSER → LM-TAD → Rule-Based Classification → Synthetic Labaled Dataset

---

# Mathematical Categorization Rules

**Formula-to-Category Mapping:**

| **Anomaly Category** | **Formula/Condition** | **Threshold Type** | **Detection Method** |
|---------------------|----------------------|-------------------|---------------------|
| **Speed Violations** | $v_{anomaly} = \begin{cases} 1 & \text{if } v_i > \mu_v + 2\sigma_v \text{ or } v_i < \mu_v - 2\sigma_v \\ 0 & \text{otherwise} \end{cases}$ | **Calculated** from data | Statistical outlier detection |
| **Route Deviation** | Path length $> NL_{value} + L_\rho$ | **Calculated** + 5km margin | Distance-based threshold |
| **Temporal Delay** | Travel time $> NT_{value} + T_\rho$ | **Calculated** + 5min margin | Time-based threshold |
| **Long Stops** | Stationary periods $> 15$ min | **Literature** (15 min) | Temporal clustering |
| **Off-Road Driving** | Distance from road $> 100$ m | **Literature** (100m) | Spatial validation |
| **U-Turn Detection** | Heading change > 150° within < 2 min | **Literature** (150°, 2min) | Angular analysis |
| **Detour** | $\text{efficiency} = \frac{\text{shortest\_path}}{\text{actual\_path}} \times \frac{\text{expected\_time}}{\text{actual\_time}} < 0.7$ | **Literature** (0.7) | Route optimization ratio |

**Symbols:** $v_i$ (velocity), $\mu_v$ (mean), $\sigma_v$ (std dev), $NL_{value}$ (normal length), $L_\rho$ (5km margin), $NT_{value}$ (normal time), $T_\rho$ (5min margin)

---



# Dataset Evaluation Strategy

**EVALUATION DATASETS:**
- **Beijing:** BJUT Private + T-Drive (public)
- **Cross-City:** Chengdu + Xi'an

**EVALUATION FRAMEWORKS:**
- **Privacy & Quality:** TransBigData + SDMetrics
- **Downstream Tasks:** LibCity Framework + Supervised Anomaly Detection
- **Analysis:** Ablation Study

**COMPREHENSIVE:** Multi-dataset, multi-metric validation

---


<!-- _class: lead -->
# Thank You
**Questions & Feedback Welcome** 
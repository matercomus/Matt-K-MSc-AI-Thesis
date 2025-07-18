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

# Dataset Characteristics & Quality

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

# Model Evaluation Process

Three candidate trajectory generation models

**EVALUATION CRITERIA:**
- Data format compatibility with GPS trajectories
- Preprocessing complexity and requirements
- Output format alignment with research needs
- Implementation feasibility and code availability

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

**ADVANTAGE:** Latest architecture with comprehensive semantic modeling

---

# HOSER Selection Justification

**INPUT REQUIREMENTS:** `id, timestamp, lon, lat, angle, speed` format

**WHY HOSER WINS:**

**Minimal Data Transformation:**
- Direct ingestion without format conversion
- No road segment mapping or network constraints (integrated in HOSER)

**Rich Metadata Utilization:**
- Leverages timestamps, speed, angle
- Preserves contextual information crucial for anomaly detection

**OUTPUT CONSISTENCY:**
- Generated trajectories match input structure
- Seamless integration with analysis pipelines

**PRACTICAL ADVANTAGES:** Fastest implementation path with lowest technical risk

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

# HOSER Performance & Advantages

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

# Anomaly Detection: Two-Step Process

**STEP 1: LM-TAD Anomaly Detection**
- **INPUT:** Synthetic trajectories from HOSER
- **PROCESS:** Language Model for Trajectory Anomaly Detection (LM-TAD)
- **METHOD:** Perplexity scoring - higher perplexity = more anomalous
- **OUTPUT:** Ranked list of potential anomalous trajectories

**STEP 2: Rule-Based Categorization**
- **INPUT:** LM-TAD identified anomalous trajectories  
- **PROCESS:** Apply mathematical rules to categorize anomaly types
- **OUTPUT:** Labeled anomaly categories with specific types

**WORKFLOW:** HOSER → LM-TAD Detection → Rule-Based Classification → Labeled Anomalies

---

# Mathematical Categorization Rules

**Distance & Efficiency Metrics:**
- **Spatial Distance:** $\text{spatialdist} = \frac{1}{e} \times \sum_{i=1}^{e} \text{dist}_i$
- **Route Efficiency:** $\text{efficiency} = \frac{\text{shortest\_path\_distance}}{\text{actual\_path\_distance}} \times \frac{\text{expected\_travel\_time}}{\text{actual\_travel\_time}}$

**Speed Anomaly Detection:**
- $v_{anomaly} = 1$ if $v_i > \mu_v + 2\sigma_v$ or $v_i < \mu_v - 2\sigma_v$

**Purpose:** These mathematical rules classify LM-TAD detected anomalies into interpretable categories

---

# Anomaly Categories & Implementation Pipeline

**ANOMALY CATEGORIES:**

**Vehicle-Based:** Speed violations (>120 km/h), Off-road driving (>100m from roads)

**Behavior-Based:** Route deviation ($> NL_{value} + L_ρ$), Temporal delay ($> NT_{value} + T_ρ$), Stop-duration (>15min), U-turns ($|Δθ| > 150°$), Detours (efficiency <0.7)

**COMPLETE PIPELINE:**
- **HOSER Generation:** Create synthetic normal trajectory baseline
- **LM-TAD Detection:** Identify anomalous trajectories using perplexity scoring
- **Rule-Based Categorization:** Apply mathematical rules to classify anomaly types
- **Iterative Refinement:** Retrain HOSER model with labeled anomalies

**NEXT STEPS:** Privacy enhancement → Complete pipeline implementation → Evaluation

---

# Questions & Discussion

**Key Decision Points:**
1. Agreement on HOSER selection for trajectory generation
2. Privacy mechanism prioritization (differential privacy vs. other approaches)
3. Evaluation metrics for synthetic data quality

**Support Needed:**
- Access to additional validation datasets (if available)
- Feedback on privacy requirements and constraints
- Guidance on thesis timeline and milestones

---

<!-- _class: lead -->
# Thank You
**Questions & Feedback Welcome** 
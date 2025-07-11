---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Privacy-Preserving Synthetic Trajectory Generation | Mateusz Kędzia'
footer: 'MSc AI Thesis | July 2025'
---

<!-- _class: lead -->
# Privacy-Preserving Synthetic Trajectory Generation for Taxi Route Anomaly Detection
## An Integrated DiffTraj-LM-TAD Framework
**Mateusz Kędzia** | MSc Artificial Intelligence | Vrije Universiteit Amsterdam
*Thesis Progress Presentation - July 2025*

---

# 1. The Research Problem

- **High Utility vs. Strong Privacy:** Trajectory data is highly useful but personally identifiable
- **Privacy-Utility Paradox:** Standard privacy methods degrade spatio-temporal patterns needed for anomaly detection
- **Research Gap:** Need for high-fidelity, private synthetic data that preserves anomaly detection utility

---

# 2. The Solution: 3-Phase Framework

![Framework Overview](mermaid-diagram.svg)

**Key Innovation:** Bootstraps anomaly generation without pre-labeled dataset

---

# Privacy by Design Principles

- **Core Innovation:** Bootstrap diverse, controllable anomalies without pre-labeled data
- **Guiding Principle:** Multiple privacy mechanisms integrated throughout pipeline

**Three Complementary Mechanisms:**
- **Differential Privacy (DP-SGD):** Formal guarantees via bounded trajectory influence
- **Trajectory-Level Protection:** Entire trajectory as atomic privacy unit  
- **Synthetic Decoupling:** New samples from learned distributions, not copies

---

# Phase 1: Baseline Generation

**Process:** Real normal data → Train DiffTraj → Generate synthetic normal data

- **Input:** Real taxi trajectory data (Beijing GPS dataset)
- **Filtering:** Extract normal trajectories using multi-criteria process:
  - Duration within 2σ of O-D medians, distance ≤ 1.5× shortest path
  - *Source:* Statistical approach following Wang et al. (2020, 2018)
- **Training:** DiffTraj (1D-CNN residual network) learns patterns from filtered real normal data
- **Generation:** Trained model produces `synthetic_normal` dataset
- **Output:** Synthetic trajectories that preserve statistical properties of real data, no direct copies

---

# Phase 2: Anomaly Mining & Curation

**Process:** Synthetic data → LM-TAD detection → Diverse querying → Rule-based curation

- **Input:** `synthetic_normal` from Phase 1
- **Detection:** LM-TAD perplexity scoring (higher = more anomalous)
- **Querying:** k-means++ diverse selection (SOEL framework)
- **Curation:** Quantitative thresholds (Wang et al. 2020, 2018):
  - Route Deviation: `Length > Normal + L_ρ`
  - Temporal Delay: `Time > Normal + T_ρ`
  - Kinematic: `Speed > 120 km/h`
- **Output:** Labeled datasets (`anomalies_speeding`, `anomalies_off_road`)

---

# Phase 3: Iterative Refinement

**Process:** Normal + Anomaly data → Retrain DiffTraj → Enhanced model

- **Input:** `synthetic_normal` + labeled anomalies from Phase 2
- **Retraining:** DiffTraj on combined dataset (5-10% anomalies)
- **Iteration:** Repeat pipeline cycles (Generate → Mine → Refine)
- **Conditional Generation:** `difftraj.sample(condition="speeding")`
- **Output:** Enhanced DiffTraj capable of targeted anomaly generation

---

# Experimental Design & Evaluation

**Cross-City Validation:**
- Primary: Beijing taxi data
- Validation: Chengdu, Xi'an (independent pipelines)

**Evaluation Framework:**
- **Anomaly Detection:** Precision, Recall, F1, AUC-PR
- **Data Quality:** Statistical fidelity, downstream tasks
- **Privacy:** Membership inference, reconstruction attacks
- **Libcity:** Evluate downstram task performance

---

# Methodological Clarifications

**Labor-Intensive Data:** Effort in curation, labeling, privacy assurance (not GPS collection)

**Non-I.I.D. Data:** Context-dependent distributions (spatial/temporal variations)

**Diffusion Privacy:** Forward noise addition → Reverse generation from new random noise

---

# Project Status & Discussion

**Current Status:**
- Phase 1: In Progress (baseline DiffTraj)
- Phase 2: Planned (LM-TAD integration)
- Phase 3: Planned (iterative refinement)

---

<!-- _class: lead -->
# Thank You
**Questions & Discussion**

---

# References & Technical Details

**Key References:**
- DiffTraj: Zhu et al. (2023)
- LM-TAD: Mbuya et al. (2024)  
- Privacy Framework: Buchholz et al. (2024)
- SOEL: Li et al. (2023)
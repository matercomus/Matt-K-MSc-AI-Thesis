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

- **Objective:** Create high-fidelity synthetic normal trajectories
- **Model:** DiffTraj (diffusion-based) for stability and quality
- **Input:** Filtered normal trajectories
  - Duration < (μ + 2σ)
  - Distance ≤ 1.5 × Shortest Path

---

# Phase 2: Anomaly Mining & Curation

- **Detection:** LM-TAD scores trajectories via perplexity (higher = more anomalous)
- **Diverse Querying:** k-means++ selects diverse anomaly candidates (SOEL framework)
- **Rule-Based Curation:**
  - Route Deviation: `Length > Normal + L_ρ`
  - Temporal Delay: `Time > Normal + T_ρ`  
  - Kinematic: `Speed > 120 km/h`
  - Off-Road: `Distance to road > threshold`

---

# Phase 3: Iterative Refinement

- **Enriched Retraining:** DiffTraj on normal + curated anomalies
- **Iterative Loop:** Generate → Mine → Refine (repeated cycles)
- **Conditional Generation:** `difftraj.sample(condition="speeding")`

---

# Experimental Design & Evaluation

**Cross-City Validation:**
- Primary: Beijing taxi data
- Validation: Chengdu, Xi'an (independent pipelines)

**Evaluation Framework:**
- **Anomaly Detection:** Precision, Recall, F1, AUC-PR
- **Data Quality:** Statistical fidelity, downstream tasks
- **Privacy:** Membership inference, reconstruction attacks

---

# Methodological Clarifications

**Labor-Intensive Data:** Effort in curation, labeling, privacy assurance (not GPS collection)

**Non-I.I.D. Data:** Context-dependent distributions (spatial/temporal variations)

**Diffusion Privacy:** Forward noise addition → Reverse generation from new random noise

**Our Approach vs. Two-Model:** Single iterative model, not separate normal/anomaly models

---

# Contributions & Impact

**Research Contributions:**
- Novel DiffTraj + LM-TAD integration
- Bootstrapping methodology without pre-labeled anomalies
- Privacy-by-design system

**Impact:**
- Academic: Reproducible privacy-preserving methodology
- Practical: Urban transport analysis without privacy compromise
- Enablement: Shareable, compliant datasets

---

# Project Status & Discussion

**Current Status:**
- Phase 1: ✅ Completed (baseline DiffTraj)
- Phase 2: 🔄 In Progress (LM-TAD integration)
- Phase 3: 📅 Planned (iterative refinement)

**Discussion Points:**
- Privacy budget allocation (ε) and thresholds (ρ)
- Three-city scope sufficiency
- Additional evaluation criteria suggestions

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

**Available:** Algorithm pseudocode, privacy calculations, detailed metrics
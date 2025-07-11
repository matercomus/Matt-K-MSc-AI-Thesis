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
**Mateusz Kędzia**
MSc Artificial Intelligence
Vrije Universiteit Amsterdam
*Thesis Progress Presentation - July 2025*
---
<!-- _class: lead -->
# 1. The Research Problem
---
## The Core Challenge in Trajectory Research
- **High Utility vs. Strong Privacy:** Trajectory data is highly useful for research but also personally identifiable.
- **The Privacy-Utility Paradox:** Standard privacy methods often degrade the spatio-temporal patterns required for anomaly detection.
- **The Research Gap:** A need for a framework to generate high-fidelity, private synthetic data that preserves utility for anomaly detection.
---
<!-- _class: lead -->
# 2. The Solution: An Integrated Framework
---
## The Core Idea: A 3-Phase Framework
![Framework Overview](mermaid-diagram.svg)
**Key Innovation:** Bootstraps anomaly generation without requiring a pre-labeled dataset.
---
## Guiding Principle: Privacy by Design
- **Differential Privacy:** DP-SGD used in training with a formal privacy budget.
  - *Example Budget:* Baseline (ε=2.0), Refinement (ε=1.0)
- **Trajectory-Level Protection:** The entire trajectory is the atomic unit of privacy, preventing correlation attacks.
- **Synthetic Data Decoupling:** The model learns statistical distributions, not copies of real data, providing inherent privacy.
---
<!-- _class: lead -->
# 3. Methodology Deep Dive
---
## Phase 1: Baseline Generation
- **Objective:** Create a high-fidelity synthetic dataset of normal trajectories.
- **Core Model:** DiffTraj (Diffusion-based) for its stability and quality.
- **Input Data:** Real trajectories filtered for normal patterns.
  - *Example Filter Criteria:*
  - Trip Duration < (μ + 2σ)
  - Trip Distance ≤ 1.5 × Shortest Path Distance
---
## Phase 2: Anomaly Mining & Curation
- **Objective:** Discover, categorize, and label anomalies from generated data.
- **Unsupervised Detection:** LM-TAD scores trajectories using a perplexity metric. Higher perplexity indicates higher anomaly likelihood.
- **Diverse Querying:** k-means++ selects a diverse subset of potential anomalies for efficient labeling (SOEL framework).
---
## Phase 2 (cont.): Rule-Based Curation
- **Objective:** Classify anomalies into interpretable categories using quantitative rules.
- **Example Rules:**
  - **Route Deviation:** `Path Length > Normal_Length + L_ρ`
  - **Temporal Delay:** `Travel Time > Normal_Time + T_ρ`
  - **Kinematic Outliers:** `Speed > 120 km/h`
  - **Off-Road:** `Distance to nearest road > threshold`
---
## Phase 3: Iterative Refinement
- **Objective:** Enhance the model's ability to create specific, labeled anomalies.
- **Enriched Retraining:** The DiffTraj model is retrained on a dataset of normal data + the curated anomalies.
- **Iterative Amplification:** The Generate → Mine → Refine loop is repeated to improve anomaly diversity and complexity.
- **Conditional Generation:** The final model can generate specific anomalies on demand.
  - *Example:* `difftraj.sample(condition="speeding")`
---
<!-- _class: lead -->
# 4. Validation & Evaluation
---
## Experimental Design
- **Primary Dataset:** Beijing taxi trajectory data.
- **Generalizability Testing:** The entire pipeline is independently run on Chengdu and Xi'an datasets.
- **Per-City Analysis:** Allows for robust evaluation of transferability across diverse urban mobility patterns.
---
## Evaluation Framework
- **Anomaly Detection Performance:**
  - Precision, Recall, F1-Score, AUC-PR
- **Synthetic Data Quality:**
  - Statistical Fidelity (e.g., Kolmogorov-Smirnov test)
  - Downstream Task Performance (e.g., travel time estimation)
- **Privacy Preservation:**
  - Membership Inference Attack simulation
  - Trajectory Reconstruction Attack simulation
---
<!-- _class: lead -->
# 5. Contribution & Impact
---
## Primary Research Contributions
- **Novel Framework:** First to integrate DiffTraj (generation) with LM-TAD (detection) for privacy-preserving anomaly synthesis.
- **Bootstrapping Methodology:** Generates labeled anomalies without pre-existing anomaly data, addressing a key research bottleneck.
- **Privacy-by-Design System:** Combines multiple, complementary privacy mechanisms for robust data protection.
---
## Expected Research Impact
- **Academic:** Provides a new, reproducible methodology for trajectory research.
- **Practical:** Enables urban transport system analysis without compromising user privacy.
- **Enablement:** Delivers shareable, privacy-compliant datasets to foster innovation.
---
<!-- _class: lead -->
# 6. Project Roadmap & Discussion
---
## Project Status & Next Steps
- **Phase 1 (Completed):** Baseline DiffTraj model is developed.
- **Phase 2 (In Progress):** LM-TAD integration and curation system development.
- **Next Steps:**
  - Complete Phase 3 (Iterative Refinement)
  - Execute Cross-City and Privacy Evaluations
  - Finalize analysis and thesis document
---
## Points for Discussion
- **Technical:** Feedback on privacy budget allocation (ε) and rule-based thresholds (ρ).
- **Methodological:** Is the three-city scope sufficient for demonstrating generalizability?
- **Scope:** Suggestions for additional evaluation criteria or downstream tasks.
---
<!-- _class: lead -->
# Thank You
**Questions & Discussion**
---
## Backup Slides
**Key References**
- **DiffTraj:** Zhu et al. (2023) - *DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model*
- **LM-TAD:** Mbuya et al. (2024) - *Trajectory Anomaly Detection with Language Models*
- **Privacy Framework:** Buchholz et al. (2024) - *A Systematisation of Knowledge for Trajectory Privacy*
- **SOEL Framework:** Li et al. (2023) - *Deep Anomaly Detection for Time Series*
**Technical Details Available**
- Algorithm Pseudocode
- Privacy Budget Calculations
- Detailed Evaluation Metrics
- Cross-City Dataset Specifications 
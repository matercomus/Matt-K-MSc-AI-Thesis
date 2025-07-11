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

**High Utility vs. Strong Privacy**
Real-world trajectory data is essential for research but is also highly sensitive. As few as four spatio-temporal points can uniquely identify an individual, exposing sensitive locations and creating a significant privacy risk.

**The Privacy-Utility Paradox**
Standard privacy-preserving methods (e.g., k-anonymity, generalization) often degrade the subtle spatio-temporal patterns that anomaly detection algorithms rely on, rendering the data less useful.

**The Research Gap**
There is a need for a framework that can generate high-fidelity synthetic trajectory data which is both useful for anomaly detection research and provides strong, verifiable privacy guarantees.

---

<!-- _class: lead -->
# 2. The Solution: An Integrated Framework

---

## The Core Idea: A 3-Phase Framework

![Framework Overview](mermaid-diagram.svg)

**Key Innovation:** The framework bootstraps the generation of diverse and controllable anomalies without requiring a pre-labeled dataset of anomalous trajectories.

---

## Guiding Principle: Privacy by Design

The framework's methodology is built upon multiple, complementary privacy protection mechanisms.

**Differential Privacy Integration**
Utilizes Differentially Private Stochastic Gradient Descent (DP-SGD) during model training to provide formal privacy guarantees.

**Trajectory-Level Protection**
Treats entire trajectories as the atomic unit of privacy, preventing correlation attacks that exploit intra-trajectory dependencies.

**Synthetic Data Decoupling**
The generative model learns statistical distributions from real data, ensuring synthetic trajectories are entirely new and not copies, which is a fundamental privacy guarantee.

---

<!-- _class: lead -->
# 3. Methodology Deep Dive

---

## Phase 1: Baseline Synthetic Data Generation

**Objective:** To create a high-fidelity synthetic dataset of normal trajectory patterns.

**Input Data**
Real-world trajectories are filtered using a multi-criteria process to isolate a dataset of verifiably normal routes.

**Core Generation Model: DiffTraj**
A diffusion-based model is used for its training stability and its proven ability to generate realistic, high-quality trajectories.

**Primary Output**
A synthetic dataset that statistically resembles the real normal data, serving as the foundation for the next phase.

---

## Phase 2: Unsupervised Anomaly Mining & Curation

**Objective:** To discover, categorize, and label anomalous patterns from the generated data.

**Unsupervised Anomaly Detection (LM-TAD)**
An autoregressive language model scores trajectories based on perplexity; higher scores indicate greater deviation from learned normal patterns.

**Diverse Querying for Labeling (SOEL-based)**
To maximize labeling efficiency, k-means++ selection is used to identify a maximally diverse subset of potential anomalies for review.

---

## Phase 2 (cont.): Rule-Based Curation

**Objective:** To classify the diverse anomalies into interpretable categories.

**Anomaly Categories**
- **Route Deviation:** Path length significantly exceeds the norm for a given origin-destination pair.
- **Temporal Delay:** Travel time is substantially longer than the typical duration.
- **Kinematic Outliers:** Trajectory violates physical or legal norms (e.g., excessive speed, unusual stationary periods).
- **Off-Road Driving:** Trajectory points deviate from the known road network.

**Result:** A curated and labeled set of interpretable synthetic anomalies.

---

## Phase 3: Iterative Refinement & Conditional Generation

**Objective:** To enrich the generative model with the ability to create specific anomalies.

**Enriched Data Retraining**
The DiffTraj model is retrained on a dataset combining the original normal data with the newly curated, labeled anomalies.

**Iterative Amplification**
The entire three-phase process can be repeated. Each cycle enhances the model's ability to generate more complex and varied anomalies.

**Controlled Anomaly Generation**
The final, refined model supports conditional sampling, enabling the targeted generation of specific anomaly types for downstream tasks.

---

<!-- _class: lead -->
# 4. Validation & Evaluation

---

## Experimental Design: Cross-City Validation

**Primary Framework Development**
The core framework is developed and optimized using the large-scale Beijing taxi dataset.

**Independent Pipeline Replication**
To assess generalizability, the entire framework is independently executed on datasets from Chengdu and Xi'an.

**Per-City Analysis**
This approach allows for a robust evaluation of framework transferability across diverse urban environments, each with unique mobility patterns.

---

## Evaluation Framework

**Anomaly Detection Performance**
Assessed using metrics suitable for imbalanced datasets, including Precision, Recall, F1-Score, and the Area Under the PR Curve (AUC-PR).

**Synthetic Data Quality**
Evaluated with the SDMetrics library, statistical distribution tests (e.g., Kolmogorov-Smirnov), and by measuring performance on downstream machine learning tasks.

**Privacy Preservation Assessment**
The strength of the privacy guarantees is tested via simulated membership inference attacks and trajectory reconstruction attacks.

---

<!-- _class: lead -->
# 5. Contribution & Impact

---

## Primary Research Contributions

**A Novel Integrated Framework**
The first to integrate a diffusion-based generative model (DiffTraj) with a language model-based anomaly detector (LM-TAD) for this purpose.

**A Bootstrapping Methodology**
A process that generates diverse, labeled anomalies without requiring pre-existing anomaly data, addressing a key bottleneck in the field.

**A Privacy-by-Design Approach**
A system that combines multiple privacy mechanisms to ensure robust protection while maintaining data utility.

---

## Expected Research Impact

**Academic Contribution**
- Provides a new methodology for privacy-preserving trajectory research.
- Establishes a framework for reproducible evaluation of anomaly detection algorithms.

**Practical Applications**
- Enables research and development for urban transport systems, including route optimization and fleet management, without compromising user privacy.

**Research Enablement**
- Delivers privacy-compliant datasets that can be shared openly, fostering further innovation in trajectory analysis.

---

<!-- _class: lead -->
# 6. Project Roadmap & Discussion

---

## Project Timeline & Current Status

**Phase 1: Baseline Generation (Completed)**
- The core DiffTraj model has been developed and trained on filtered normal trajectory data.

**Phase 2: Anomaly Mining (In Progress)**
- Integration of the LM-TAD model and implementation of the rule-based curation system is underway.

**Phase 3: Iterative Refinement (Planned)**
- The iterative retraining loop and conditional generation capabilities are the next development steps.

---

## Remaining Milestones

**Cross-City Validation**
- Execute the full pipeline on the Chengdu and Xi'an datasets to evaluate model generalizability.

**Privacy Evaluation**
- Conduct rigorous attack simulations to quantify the privacy-utility trade-off of the generated datasets.

**Thesis Completion (Target: June 2025)**
- Complete the final analysis of results, write the discussion and conclusion, and prepare the final document.

---

## Points for Discussion

**Technical & Methodological**
1.  Are the proposed privacy budget allocations for differential privacy appropriate?
2.  Feedback on the rule-based curation system for anomaly categorization.
3.  Is the three-city scope sufficient for demonstrating framework generalizability?

**Research Scope & Direction**
1.  Considering the timeline, are there areas to prioritize or de-scope?
2.  Suggestions for additional evaluation criteria or downstream tasks.
3.  Potential target journals or conferences for publication.

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
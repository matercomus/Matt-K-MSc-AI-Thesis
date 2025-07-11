# Privacy-Preserving Synthetic Trajectory Generation for Taxi Route Anomaly Detection
## Methodology Overview - Thesis Coordinator Presentation

**Author:** Mateusz Kędzia  
**Date:** July 2025  
**Thesis Title:** Privacy-Preserving Synthetic Trajectory Generation for Taxi Route Anomaly Detection: An Integrated DiffTraj-LM-TAD Framework

---

## 📋 Executive Summary

### Research Problem
- **Challenge:** Real taxi trajectory data contains sensitive location information that limits research access
- **Gap:** Traditional privacy methods destroy spatio-temporal patterns needed for anomaly detection
- **Need:** Framework for generating synthetic taxi trajectories that preserve anomaly detection utility while ensuring privacy

### Solution Approach
**Integrated 3-Phase Framework** combining:
- **DiffTraj** (diffusion-based generation) + **LM-TAD** (language model anomaly detection)
- Iterative refinement process to bootstrap anomaly generation without pre-labeled data
- Multiple privacy protection mechanisms integrated throughout pipeline

---

## 🔬 Core Methodology Framework

### Phase 1: Baseline Synthetic Data Generation
```
Real Normal Trajectories → DiffTraj Training → Synthetic Normal Dataset
```

**Key Components:**
- **Model:** DiffTraj (1D-CNN-based diffusion model)
- **Input:** Filtered normal taxi trajectories (statistical thresholds applied)
- **Training:** Multi-criteria filtering (trip duration, distance efficiency, temporal patterns)
- **Output:** High-fidelity synthetic normal trajectories with privacy by design

**Rationale:** 
- Diffusion models provide superior training stability vs GANs
- No direct copying ensures inherent privacy protection
- Foundation dataset for anomaly mining process

### Phase 2: Unsupervised Anomaly Mining and Curation
```
Synthetic Data → LM-TAD Detection → Diverse Querying → Rule-Based Curation → Labeled Anomalies
```

**Key Components:**

1. **LM-TAD Anomaly Detection**
   - Autoregressive transformer treating trajectories as token sequences
   - Perplexity-based anomaly scoring (higher perplexity = more anomalous)
   - No pre-labeled anomaly data required

2. **Diverse Querying (k-means++ Selection)**
   - Based on SOEL framework for maximum labeling efficiency
   - Selects maximally diverse trajectories in feature space
   - Focuses manual review effort on representative samples

3. **Rule-Based Categorization**
   - Route Deviation: Path length > Normal + threshold
   - Temporal Delay: Travel time > Normal + threshold  
   - Kinematic Outliers: Speed/behavior violations
   - Off-Road Driving: GPS points away from road network

**Output:** Categorized anomaly datasets (speeding, off-road, route deviation, etc.)

### Phase 3: Iterative Refinement and Conditional Generation
```
Normal + Anomalies → Enriched Training → Refined DiffTraj → Conditional Generation
```

**Key Components:**
- **Enriched Retraining:** DiffTraj trained on combined normal + curated anomalies
- **Iterative Amplification:** Multiple cycles of generation → detection → curation → retraining
- **Controlled Generation:** Final model supports conditional sampling by anomaly type

**Benefits:**
- Progressively improves anomaly diversity and quality
- Enables targeted generation of specific anomaly types
- Maintains balance to prevent mode collapse

---

## 🔒 Privacy Protection Strategy

### Multi-Layer Privacy Approach

1. **Differential Privacy Integration**
   - DP-SGD during DiffTraj training
   - Bounded individual trajectory influence
   - Privacy budget allocation: ε₁=2.0 (baseline), ε₂=1.0 (refinement), ε₃=0.5 (evaluation)

2. **Trajectory-Level Protection**
   - Entire trajectories as atomic privacy units
   - Prevents correlation attacks within trajectories
   - Spatial generalization through grid-based discretization

3. **Synthetic Data Decoupling**
   - Generated from learned distributions, not direct copies
   - No individual trajectory reconstruction possible
   - Fundamental privacy guarantee through generative approach

4. **Attack Resistance**
   - Defense against membership inference attacks
   - Protection against reconstruction attacks
   - Iterative refinement diminishes individual influence

---

## 🧪 Experimental Design

### Validation Strategy
- **Primary Development:** Beijing taxi dataset (Nov 25 - Dec 1, 2019)
- **Cross-City Validation:** Independent pipeline execution on Chengdu and Xi'an datasets
- **Per-City Approach:** Separate models, privacy budgets, and training for each city

### Evaluation Framework

**1. Anomaly Detection Performance**
- Precision, Recall, F1-Score
- AUC-ROC and AUC-PR curves
- Performance across different anomaly categories

**2. Synthetic Data Quality**
- SDMetrics standardized assessment (Resemblance, Utility, Privacy)
- Statistical fidelity (KS-test, Jensen-Shannon divergence)
- Downstream task performance validation

**3. Privacy Preservation Assessment**
- Membership inference attack resistance
- Trajectory reconstruction attack evaluation
- Privacy-utility trade-off quantification

---

## 🎯 Key Innovations

### Technical Contributions
1. **Novel Integration:** First framework combining DiffTraj generation with LM-TAD detection
2. **Bootstrapping Approach:** Generates diverse anomalies without pre-labeled data
3. **Privacy-by-Design:** Multiple complementary privacy mechanisms integrated throughout
4. **Interpretable Anomalies:** Rule-based curation ensures controllable, categorized outputs

### Methodological Advantages
- **No Labeled Data Required:** Addresses common bottleneck in anomaly detection research
- **High Control:** Rule-based curation provides interpretable anomaly categories
- **Scalable:** Computationally efficient compared to complex latent space methods
- **Generalizable:** Cross-city validation demonstrates urban environment adaptability

---

## 📈 Expected Outcomes

### Primary Deliverables
1. **Synthetic Dataset:** Privacy-preserving taxi trajectory dataset with controlled anomalies
2. **Framework Implementation:** Complete pipeline for privacy-preserving trajectory generation
3. **Evaluation Results:** Comprehensive assessment across three urban environments
4. **Privacy Analysis:** Quantified trade-offs between privacy protection and utility

### Research Impact
- **Academic:** Novel methodology for privacy-preserving trajectory research
- **Practical:** Enables anomaly detection research without sensitive data access
- **Regulatory:** Framework addresses data protection requirements for transportation research

---

## 🔄 Implementation Timeline

1. **Phase 1 Implementation** ✅ (Completed)
   - DiffTraj baseline model development
   - Normal trajectory filtering and training

2. **Phase 2 Development** 🔄 (In Progress)
   - LM-TAD integration and anomaly mining
   - Rule-based curation system implementation

3. **Phase 3 Integration** 📅 (Planned)
   - Iterative refinement framework
   - Conditional generation capabilities

4. **Cross-City Validation** 📅 (Planned)
   - Chengdu and Xi'an dataset evaluation
   - Generalizability assessment

5. **Privacy Evaluation** 📅 (Planned)
   - Attack simulation and resistance testing
   - Privacy-utility trade-off analysis

---

## ❓ Discussion Points for Coordinators

### Technical Questions
1. **Model Integration:** Feedback on DiffTraj-LM-TAD compatibility approach
2. **Privacy Budget Allocation:** Appropriateness of differential privacy parameter choices
3. **Evaluation Metrics:** Adequacy of proposed privacy-utility assessment framework

### Methodological Considerations
1. **Iterative Approach:** Trade-offs between complexity and performance gains
2. **Rule-Based Curation:** Balance between automation and manual oversight
3. **Cross-City Validation:** Sufficiency of three-city evaluation scope

### Research Scope
1. **Timeline Feasibility:** Realistic completion schedule for remaining phases
2. **Resource Requirements:** Computational and data access needs
3. **Publication Strategy:** Conference/journal targets for methodology and results

---

## 📚 Key References

- **DiffTraj:** Zhu et al. (2023) - "DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model"
- **LM-TAD:** Mbuya et al. (2024) - "Trajectory Anomaly Detection with Language Models"
- **Privacy Framework:** Buchholz et al. (2024) - "A Systematisation of Knowledge for Trajectory Privacy"
- **SOEL Framework:** Li et al. (2023) - "Deep Anomaly Detection for Time Series" 
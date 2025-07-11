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

## The Core Challenge

🚕 **Real taxi trajectory data** contains sensitive location information
- As few as **4 spatio-temporal points** can identify 95% of individuals
- Reveals home, work, and personal POIs
- Creates barriers for anomaly detection research

🔒 **Traditional privacy methods** destroy essential patterns
- k-anonymity, differential privacy degrade spatio-temporal relationships
- Anomaly detection algorithms require these exact patterns
- **Privacy vs. Utility trade-off**

📊 **Research gap:** No framework exists for privacy-preserving trajectory anomaly generation

---

<!-- _class: lead -->
# 2. The Solution: An Integrated Framework

---

## The Core Idea: A 3-Phase Framework

![Framework Overview](mermaid-diagram.svg)

**Key Innovation:** Bootstraps anomaly generation without pre-labeled data

---

## Guiding Principle: Privacy by Design

### 🔐 **Differential Privacy Integration**
- DP-SGD during training
- Privacy budget: ε₁=2.0, ε₂=1.0, ε₃=0.5

### 🛡️ **Trajectory-Level Protection**
- Entire trajectories as atomic units
- Prevents correlation attacks

### 🔀 **Synthetic Data Decoupling**
- Generated from learned distributions
- No individual reconstruction possible

### ⚔️ **Attack Resistance**
- Defense against membership inference
- Protection against reconstruction attacks

---

<!-- _class: lead -->
# 3. Methodology Deep Dive

---

## Phase 1: Baseline Synthetic Data Generation

### Input: Filtered Normal Trajectories
- Multi-criteria filtering (duration, distance, temporal patterns)
- Statistical thresholds applied
- Clean foundation dataset

### Model: DiffTraj
- **1D-CNN-based diffusion model**
- Superior training stability vs GANs
- No direct copying → **inherent privacy protection**

### Output: High-fidelity synthetic normal trajectories

---

## Phase 2: Unsupervised Anomaly Mining

### 🤖 LM-TAD Anomaly Detection
- Autoregressive transformer treating trajectories as token sequences
- **Perplexity-based scoring:** Higher perplexity = more anomalous
- No pre-labeled data required

### 🎯 Diverse Querying (k-means++ Selection)
- Based on SOEL framework
- Maximally diverse trajectories in feature space
- Efficient manual review

---

## Phase 2 (cont.): Rule-Based Categorization

### 📏 **Route Deviation**
Path length > Normal + threshold (e.g., +5km)

### ⏱️ **Temporal Delay** 
Travel time > Normal + threshold (e.g., +5min)

### 🚗 **Kinematic Outliers**
Speed violations, unusual stopping patterns

### 🛣️ **Off-Road Driving**
GPS points away from road network

**Result:** Categorized, interpretable anomaly datasets

---

## Phase 3: Iterative Refinement & Conditional Generation

### 🔄 Enriched Retraining
- DiffTraj trained on Normal + Curated Anomalies
- Careful balance to prevent mode collapse

### 📈 Iterative Amplification
- Multiple cycles: Generate → Detect → Curate → Retrain
- Progressive improvement in anomaly diversity

### 🎛️ Controlled Generation
- Final model supports conditional sampling
- `difftraj.sample(condition="speeding")`

---

<!-- _class: lead -->
# 4. Validation & Evaluation

---

## Experimental Design: Cross-City Validation

### 🏙️ **Primary Development**
Beijing taxi dataset (Nov 25 - Dec 1, 2019)

### 🌏 **Independent Validation**
- **Chengdu** and **Xi'an** datasets
- Separate models, privacy budgets, training
- Tests generalizability across urban environments

### 📊 **Per-City Approach**
- Fair comparison across different mobility patterns
- No joint multi-city training complexity

---

## Evaluation Framework

### 🎯 **Anomaly Detection Performance**
- Precision, Recall, F1-Score
- AUC-ROC and AUC-PR curves
- Category-specific performance

### 📈 **Synthetic Data Quality**
- SDMetrics assessment (Resemblance, Utility, Privacy)
- Statistical fidelity (KS-test, Jensen-Shannon)
- Downstream task validation

### 🔒 **Privacy Preservation**
- Membership inference attack resistance
- Trajectory reconstruction evaluation
- Privacy-utility trade-off quantification

---

<!-- _class: lead -->
# 5. Contribution & Impact

---

## Technical Contributions

### 🆕 **Novel Integration**
First framework combining DiffTraj generation with LM-TAD detection

### 🔄 **Bootstrapping Approach** 
Generates diverse anomalies without pre-labeled data

### 🔐 **Privacy-by-Design**
Multiple complementary mechanisms integrated throughout

### 🎯 **Interpretable Anomalies**
Rule-based curation ensures controllable outputs

---

## Methodological Advantages

✅ **No Labeled Data Required** - Addresses research bottleneck  
✅ **High Control** - Interpretable anomaly categories  
✅ **Scalable** - Computationally efficient approach  
✅ **Generalizable** - Cross-city validation demonstrates adaptability  

---

## Expected Research Impact

### 🎓 **Academic Impact**
- Novel methodology for privacy-preserving trajectory research
- First integration of diffusion models with language model anomaly detection
- Framework for reproducible anomaly detection evaluation

### 🏢 **Practical Applications**
- Urban transportation anomaly detection
- Ride-sharing route optimization
- Taxi fleet management systems

### 📊 **Research Enablement**
- Privacy-compliant datasets for trajectory analysis
- Addresses data protection requirements

---

<!-- _class: lead -->
# 6. Project Roadmap & Discussion

---

## Timeline & Current Status

### ✅ **Phase 1 Implementation** (Completed)
- DiffTraj baseline model development
- Normal trajectory filtering and training

### 🔄 **Phase 2 Development** (In Progress)
- LM-TAD integration and anomaly mining
- Rule-based curation system implementation

### 📅 **Phase 3 Integration** (Planned)
- Iterative refinement framework
- Conditional generation capabilities

---

## Remaining Milestones

### 📅 **Cross-City Validation** (Planned)
- Chengdu and Xi'an dataset evaluation
- Generalizability assessment

### 📅 **Privacy Evaluation** (Planned)
- Attack simulation and resistance testing
- Privacy-utility trade-off analysis

### 📝 **Thesis Completion** (Target: June 2025)
- Results analysis and writing
- Final evaluation and discussion

---

## Questions for Coordinators

### 🔧 **Technical Questions**
1. **Model Integration:** Feedback on DiffTraj-LM-TAD compatibility
2. **Privacy Budget:** Appropriateness of DP parameter choices
3. **Evaluation Metrics:** Adequacy of assessment framework

### 🎯 **Methodological Considerations**
1. **Iterative Approach:** Complexity vs. performance trade-offs
2. **Rule-Based Curation:** Automation vs. manual oversight balance
3. **Cross-City Scope:** Sufficiency of three-city evaluation

---

<!-- _class: lead -->
# Thank You & Questions

**Thank you for your attention!**

📧 **Contact:** m.k.kedzia@student.vu.nl  
📄 **Thesis Repository:** [Available upon request]  
🔗 **Literature Review:** Comprehensive analysis of 50+ papers

---

## Backup Slides

### Key References
- **DiffTraj:** Zhu et al. (2023) - Diffusion Probabilistic Model for GPS Trajectories
- **LM-TAD:** Mbuya et al. (2024) - Language Models for Trajectory Anomaly Detection  
- **Privacy Framework:** Buchholz et al. (2024) - Trajectory Privacy Systematisation
- **SOEL Framework:** Li et al. (2023) - Deep Anomaly Detection for Time Series

### Technical Details Available
- Algorithm pseudocode
- Privacy budget calculations
- Detailed evaluation metrics
- Cross-city dataset specifications 
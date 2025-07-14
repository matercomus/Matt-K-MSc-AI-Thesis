---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Privacy-Preserving Synthetic Trajectory Generation | Mateusz Kędzia'
footer: 'MSc AI Thesis | July 2025'
---

<!-- _class: lead -->
# **Privacy-Preserving Synthetic Trajectory Generation for Taxi Route Anomaly Detection**
## *An Integrated DiffTraj-LM-TAD Framework*

**Mateusz Kędzia** | MSc Artificial Intelligence | Vrije Universiteit Amsterdam  
*Thesis Progress Presentation - July 2025*

---

# **The Core Challenge**

### **Privacy vs. Utility Paradox**
- **High Utility Data**: Trajectory data is incredibly valuable for research
- **High Privacy Risk**: Personally identifiable location patterns
- **Traditional Trade-off**: Privacy protection destroys spatio-temporal patterns

> *"Standard privacy methods degrade spatio-temporal patterns needed for anomaly detection"*

---

# **Research Gap**

### **Key Problems:**
- **Data Scarcity**: Limited access to real trajectory datasets
- **Privacy Barriers**: Sensitive location information blocks research
- **Detection Limitations**: Need preserved patterns for effective anomaly detection

### **Our Innovation**
**Privacy by Design** + **High Utility** + **Anomaly Detection Ready**

---

# **The Solution: 3-Phase Framework**

### **Framework Overview**
```
Phase 1: Normal Baseline → Phase 2: Anomaly Mining → Phase 3: Enhanced Model
```

| Phase | Input | Output |
|-------|-------|---------|
| **1** | Real normal data | `synthetic_normal` |
| **2** | Synthetic normal | Labeled anomalies |
| **3** | Normal + Anomalies | Enhanced model |

---

# **Key Innovation**

### **Bootstrap Approach**
> *"Bootstraps anomaly generation without pre-labeled dataset"*

**Why This Works:**
- **Iterative Learning**: Each cycle improves understanding
- **Controlled Generation**: Rule-based curation
- **Privacy First**: From distributions, not copies

---

# **Core Model 1: DiffTraj**

### **Diffusion-Based Generation**
> *"Diffusion Probabilistic Model for GPS Trajectory Generation"*

**Key Process:**
```
White Noise → Reverse Denoising → Realistic Trajectory
```

**Architecture:**
- 1D-CNN-based residual network with attention
- Works directly with continuous GPS coordinates

---

# **Why DiffTraj?**

| Feature | **DiffTraj** | GANs | VAEs |
|---------|-------------|------|------|
| **Training Stability** | **Excellent** | Mode collapse | Good |
| **Sample Quality** | **High-fidelity** | Good | Blurry |
| **Privacy Protection** | **Inherent** | Risk | Risk |
| **Controllability** | **Conditional** | Limited | Good |

---

# **Core Model 2: LM-TAD**

### **Language Model for Trajectories**
> *"Treats trajectories as sequences of tokens"*

**Process:**
```
GPS Trajectory → Token Sequence → Transformer → Perplexity Score
```

**Key Features:**
- **Autoregressive**: Predicts next location
- **Perplexity Scoring**: Higher = more anomalous
- **Interpretable**: Clear anomaly explanations

---

# **Why LM-TAD?**

### **Perfect for Bootstrapping**
**No Labeled Data Required**  
**Interpretable Scores**  
**Online Detection Capability**  
**GPS Compatible**

### **State-of-the-Art Performance**
- Demonstrated on synthetic and real-world datasets
- User-specific anomaly detection
- Efficient attention caching

---

# **Core Framework: SOEL**

### **Smart Anomaly Selection**
> *"Selecting informative data points for expert feedback"*

**Process:**
```
Feature Extraction → k-means++ → Diverse Candidates
```

**Benefits:**
- **Optimal Coverage**: Maximizes diversity
- **Labeling Efficiency**: Most information from limited labels
- **Proven Theory**: Conditions for generalization

---

# **Privacy by Design**

### **The Challenge**
> *"Traditional privacy methods destroy spatio-temporal relationships"*

### **Our Three-Layer Defense**
1. **Differential Privacy**: DP-SGD with privacy budgets
2. **Trajectory-Level Protection**: Entire trajectory as unit
3. **Synthetic Decoupling**: Generation from distributions

---

# **Layer 1: Differential Privacy**

### **DP-SGD Integration**
- **Privacy Budgets**: ε₁=2.0, ε₂=1.0, ε₃=0.5
- **Bounded Influence**: Individual trajectory impact limited
- **Source**: PATE-GAN principles

### **Protection Mechanism**
- Noise addition during training
- Prevents membership inference attacks

---

# **Layer 2: Trajectory-Level Protection**

### **Atomic Privacy Units**
- **Approach**: Entire trajectory as privacy unit
- **Benefit**: Stops correlation attacks
- **Proven**: Optimal approach (Buchholz et al., 2024)

### **Attack Prevention**
- Prevents temporal dependency exploitation
- Maintains spatial pattern integrity

---

# **Layer 3: Synthetic Decoupling**

### **Core Innovation**
- **Generation**: From learned distributions, not copies
- **Inherent Protection**: No direct relationship to real data
- **Foundation**: Generative privacy theory

### **Privacy Guarantee**
- Synthetic trajectories ≠ real trajectory transformations
- Novel patterns not in original dataset

---

# **Phase 1: Baseline Generation**

### **Building the Foundation**
```
Real Normal Data → Train DiffTraj → Synthetic Normal Data
```

### **Smart Filtering**
- **Duration**: Within 2σ of O-D medians
- **Distance**: ≤ 1.5× shortest path
- **Temporal**: Typical patterns only

---

# **Phase 2: Anomaly Mining**

### **Unsupervised Detection Process**
```
synthetic_normal → LM-TAD → Perplexity Scoring
```

### **Step 1: LM-TAD Scoring**
- **Input**: `synthetic_normal` trajectories
- **Process**: Autoregressive language modeling
- **Output**: Perplexity scores (higher = more anomalous)

---

# **Phase 2: Diverse Selection**

### **k-means++ Selection**
- **Purpose**: Maximize labeling efficiency
- **Method**: SOEL framework implementation
- **Result**: Diverse anomaly candidates

### **Rule-Based Curation**
- **Route Deviation**: `Length > Normal + L_ρ`
- **Temporal Delay**: `Time > Normal + T_ρ`
- **Kinematic**: `Speed > 120 km/h`

---

# **Phase 3: Iterative Refinement**

### **Closing the Loop**
```
Normal + Anomalies → Retrain DiffTraj → Enhanced Model
```

### **Process Components**
- **Enriched Training**: 5-10% anomalies
- **Iterative Cycles**: Generate → Mine → Refine
- **Conditional Generation**: `difftraj.sample(condition="speeding")`

---

# **Experimental Design**

### **Cross-City Validation**
| Dataset | Role | Purpose |
|---------|------|---------|
| **Beijing** | Primary | Framework development |
| **Chengdu** | Validation | Cross-city generalization |
| **Xi'an** | Validation | Urban diversity testing |

---

# **Evaluation Framework**

### **Three-Dimensional Assessment**

**1. Anomaly Detection Performance**
- Precision, Recall, F1, AUC-PR

**2. Data Quality Assessment**
- SDMetrics (Resemblance, Utility, Privacy)

**3. Privacy Protection**
- Membership inference attacks
- Reconstruction attacks

---

# **Why This Methodology Works**

### **Model Choice Rationale**

**DiffTraj Advantages:**
- Proven trajectory generation superiority
- Inherent privacy protection
- Conditional generation support

**LM-TAD Integration:**
- No labeled data required
- Interpretable anomaly scores
- Bootstrap-friendly approach

---

# **Key Contributions**

### **Technical Innovations**
1. **Integrated Framework**: First DiffTraj + LM-TAD combination
2. **Bootstrap Methodology**: Anomaly generation without labels
3. **Multi-Layer Privacy**: Comprehensive protection strategy

### **Research Impact**
- Novel privacy-preserving trajectory research
- Reproducible evaluation framework
- Open-source implementation

---

# **Real-World Applications**

### **Practical Use Cases**
- **Taxi Fleet Management**: Route optimization
- **Urban Planning**: Traffic pattern analysis
- **Privacy Research**: Synthetic data generation
- **Anomaly Detection**: System development

---

# **Current Status**

### **Implementation Progress**
- **Phase 1**: In Progress (baseline DiffTraj)
- **Phase 2**: Planned (LM-TAD integration)
- **Phase 3**: Planned (iterative refinement)

### **Next Steps**
1. Complete Phase 1 baseline
2. Implement LM-TAD integration
3. Develop rule-based curation
4. Cross-city validation

---

# **Expected Outcomes**

### **Deliverables**
- **Synthetic Datasets**: Privacy-preserving anomaly data
- **Open Framework**: Reproducible research pipeline
- **Evaluation Results**: Comprehensive performance analysis
- **Research Publication**: Novel methodology contribution

---

<!-- _class: lead -->
# **Thank You**
## **Questions & Discussion**

### **Key Takeaways**
**Privacy-Preserving**: Multi-layer protection  
**Bootstrap Approach**: No labeled data required  
**Practical Framework**: Real-world applications  
**Open Research**: Reproducible and extensible

---

# **References**

### **Core References**
- **DiffTraj**: Zhu et al. (2023)
- **LM-TAD**: Mbuya et al. (2024)
- **SOEL**: Li et al. (2023)
- **Privacy Framework**: Buchholz et al. (2024)

### **Technical Implementation**
- **DiffTraj**: 1D-CNN residual network
- **LM-TAD**: Autoregressive transformer
- **Privacy**: DP-SGD + trajectory-level protection 
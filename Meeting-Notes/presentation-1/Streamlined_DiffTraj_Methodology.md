# **Synthetic Anomalous Taxi Trajectory Generation: A DiffTraj-Based Methodology**
*Master's Thesis Implementation Plan - Presentation*

---

## **Problem Statement & Motivation**

### **The Urban Mobility Challenge**
- **Data Scarcity**: Anomalous taxi events are rare (~1-5% of all trajectories) [[Anomalous_Taxi_Trajectory_Generation_.md#addressing-data-scarcity-and-imbalance]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#addressing-data-scarcity-and-imbalance)
- **Privacy Concerns**: Real GPS trajectories contain sensitive personal information [[Methodology_Anomalous_Taxi_Trajectory_Generation_.md#6-privacy-considerations]](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md#6-privacy-considerations)
- **Detection Limitations**: Traditional anomaly detection lacks sufficient training data

### **Why Synthetic Data?**
> *"Synthetic data directly addresses these challenges by enabling the controlled generation of both normal and anomalous samples, tailored to specific applications"* [[Anomalous_Taxi_Trajectory_Generation_.md#3-motivations-and-benefits]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#3-motivations-and-benefits)

**Key Benefits:**
- **Scalable**: Generate unlimited anomalous samples
- **Privacy-Preserving**: No real individual trajectories exposed
- **Controllable**: Define specific anomaly types and intensities

---

## **What Are Anomalous Taxi Trajectories?**

### **Beyond Statistical Outliers**
> *"Anomalous taxi trajectories are not merely statistical outliers but are defined as deviations from typical or regular driving patterns"* [[Anomalous_Taxi_Trajectory_Generation_.md#2-introduction]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#2-introduction)

### **Two Primary Categories:**

#### **Subjective Anomalies** (Intentional)
- **Taxi Fraud**: Unnecessary detours to overcharge passengers
- **Route Manipulation**: Deliberately longer paths
- **Characteristic**: Significant increase in trajectory length

#### **Objective Anomalies** (Environmental)
- **Traffic Events**: Accidents, congestion, road closures
- **Characteristic**: Increased travel time, normal distance

### **Four Behavior Patterns (Abp1-4)**
> *Reference: [Methodology_Anomalous_Taxi_Trajectory_Generation_.md#3.1-defining-anomalies](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md#31-defining-anomalies-in-taxi-trajectories)*

| Pattern | Length | Time | Interpretation |
|---------|--------|------|----------------|
| **Abp1** | Normal | Normal | Efficient route/smooth traffic |
| **Abp2** | Normal | **High** | Traffic congestion/delays |
| **Abp3** | **High** | Normal | Strategic detour, good timing |
| **Abp4** | **High** | **High** | **Fraudulent driving** |

---

## **Why DiffTraj? Technical Foundation**

### **Diffusion Models for Trajectory Generation**
> *"DMs excel at capturing complex data distributions and generating high-fidelity samples, making them particularly well-suited for modeling the stochastic and intricate characteristics inherent in human behavior"* [[Methodology_Anomalous_Taxi_Trajectory_Generation_.md#1-introduction]](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md#1-introduction)

### **DiffTraj Architecture**
**Two-Stage Process:**
- **Forward Process**: Real Trajectory → (+Noise) → White Noise  
- **Reverse Process**: White Noise → (-Noise) → Synthetic Trajectory

#### **Key Components:**
- **Traj-UNet**: 1D-CNN-based residual network with attention
- **Conditional Generation**: Incorporates velocity, distance, time, regions
- **Privacy by Design**: Generates from noise, never replicates real trajectories

### **Advantages Over Alternatives**
> *Reference: [Anomalous_Taxi_Trajectory_Generation_.md#generative-models-for-synthetic-trajectory-generation](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#generative-models-for-synthetic-trajectory-generation)*

| Feature | DiffTraj | GANs | VAEs |
|---------|----------|------|------|
| **Training Stability** | ✅ High | ⚠️ Unstable | ✅ Stable |
| **Sample Quality** | ✅ High | ✅ High | ⚠️ Blurry |
| **Privacy Preservation** | ✅ Inherent | ❌ Risk | ❌ Risk |
| **Controllability** | ✅ High | ⚠️ Limited | ✅ Good |

---

## **Our Multi-Dataset Approach**

### **Cross-City Validation Strategy**
*Comprehensive evaluation across diverse urban environments*

#### **Available Datasets:**
1. **Beijing T-Drive** (Public)
   - 10,357 taxis, 15M GPS points, 1 week [[Anomalous_Taxi_Trajectory_Generation_.md#key-datasets-and-their-characteristics]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#key-datasets-and-their-characteristics)
   - Established DiffTraj benchmark
   
2. **Chengdu & Xi'an** (DiffTraj Paper)
   - Additional Chinese cities for generalization testing
   
3. **Porto Taxi** (Potential)
   - European urban patterns for international validation
   
4. **BJUT Private Beijing** (Private)
   - Unique validation against public Beijing data

### **Research Advantage**
> *"This multi-dataset strategy enables improved generalization of anomaly patterns, enhanced research impact, and risk mitigation by having multiple datasets to rely on"*

**Cross-City Benefits:**
- **Generalization**: Test if patterns transfer between cities
- **Robustness**: Validate across different urban structures
- **Comparison**: Public vs. private dataset performance analysis

---

## **Methodology Overview - 3-Month Timeline**

### **Streamlined Approach: Rule-Based Focus**
> *"Timeline Constraint: 3 months (July 8 - October 8, 2025) - focus on rule-based methods only, leave advanced techniques for future work"*

```
Month 1: Foundation → Month 2: Core Contribution → Month 3: Integration
```

#### **Why Rule-Based for Phase 1?**
- **Feasibility**: Implementable within 3-month constraint
- **Control**: Full control over anomaly characteristics
- **Interpretability**: Clear understanding of generated patterns
- **Foundation**: Establishes baseline for future advanced methods

> *Reference to advanced techniques: [Methodology_Anomalous_Taxi_Trajectory_Generation_.md#3.2-techniques](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md#32-techniques-for-controlled-anomaly-injection-via-diffusion-models)*

---

## **Month 1 - Foundation Setup (Jul 8 - Aug 8)**

### **Parallel Implementation Strategy**

#### **Week 1-2: DiffTraj & Detection Setup**
**Concurrent Development:**
- **DiffTraj Implementation**: GitHub code adaptation, multi-dataset preprocessing, initial model training
- **Anomaly Detection Baseline**: Isolation Forest implementation, basic feature extraction, performance benchmarking

#### **Week 3-4: Manual Labeling & Validation**
- **Ground Truth Creation**: 50-100 trajectories per dataset (manual labeling for validation)
- **Abp1-4 Classification**: Manual categorization using behavior patterns
- **Cross-Dataset Validation**: Consistency testing

### **Evaluation Metrics (Month 1)**
- **Visual Inspection**: Trajectory plotting and pattern recognition
- **ROC-AUC**: Basic anomaly detection performance
- **Statistical Comparison**: Generated vs. real trajectory distributions

---

## **Month 2 - Anomaly Generation (Aug 8 - Sep 8)**
*CORE RESEARCH CONTRIBUTION*

### **Rule-Based Anomaly Injection Framework**

#### **Taxonomy Implementation**
> *Based on [Methodology_Anomalous_Taxi_Trajectory_Generation_.md - Table 2](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md#31-defining-anomalies-in-taxi-trajectories)*

**Anomaly Injection Methods:**
- **Spatial Deviations**: Route deviations and circuitous paths
- **Temporal Anomalies**: Duration extensions and traffic delays  
- **Kinematic Outliers**: Speed violations and abrupt changes
- **Hybrid Anomalies**: Combined anomaly types

#### **Specific Implementation Targets**
1. **Route Deviations**: Insert detours increasing distance by 20-50% [[Anomalous_Taxi_Trajectory_Generation_.md#rule-based-perturbation-methods]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#rule-based-perturbation-methods)
2. **Speed Anomalies**: Modify velocity profiles (excessive speed/sudden stops) [[Anomalous_Taxi_Trajectory_Generation_.md#rule-based-perturbation-methods]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#rule-based-perturbation-methods)
3. **Stop Anomalies**: Add unexpected or extended stops (5-30 minutes) [[Methodology_Anomalous_Taxi_Trajectory_Generation_.md#31-defining-anomalies-in-taxi-trajectories]](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md#31-defining-anomalies-in-taxi-trajectories)
4. **Duration Extensions**: Simulate traffic delays or fraudulent driving [[Anomalous_Taxi_Trajectory_Generation_.md#defining-anomalies-in-taxi-trajectories-characteristics-and-types]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#defining-anomalies-in-taxi-trajectories-characteristics-and-types)

#### **Week-by-Week Progress**
- **Week 1**: Basic injection algorithms for all 4 Abp patterns
- **Week 2**: Pattern-specific generation (Abp1: efficient, Abp4: fraudulent)
- **Week 3**: Quality testing and IForest validation
- **Week 4**: Diversity enhancement and cross-city testing

---

## **Month 3 - Integration & Evaluation (Sep 8 - Oct 8)**

### **Comprehensive Evaluation Framework**

#### **Multi-Dimensional Assessment**
> *Reference: [Methodology_Anomalous_Taxi_Trajectory_Generation_.md#5-evaluation](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md#5-evaluation-framework)*

**Evaluation Components:**
- **Synthetic Quality Assessment**: SDMetrics analysis (resemblance, utility, privacy), LibCity downstream task performance, statistical similarity tests
- **Detection Performance**: Precision, Recall, F1-Score, ROC-AUC, cross-city generalization testing, Abp1-4 pattern-specific metrics  
- **Cross-Dataset Validation**: Beijing public vs. private comparison, inter-city pattern transfer analysis, statistical significance testing

#### **Implementation Schedule**
- **Week 1**: End-to-end pipeline integration
- **Week 2**: Comprehensive metric evaluation
- **Week 3**: Cross-city and cross-dataset analysis
- **Week 4**: Documentation and thesis writing

---

## **Technical Implementation Details**

### **Core Components**
- **Base Trajectory Generation**: DiffTraj model adaptation and training
- **Anomaly Injection**: Rule-based perturbation methods
- **Detection System**: Isolation Forest baseline with custom metrics
- **Evaluation Framework**: SDMetrics for synthetic data quality, cross-city validation

### **Key Tools & Dependencies**
- **DiffTraj**: PyTorch implementation from [GitHub](https://github.com/Yasoz/DiffTraj)
- **Detection**: scikit-learn (Isolation Forest), custom evaluation metrics
- **Evaluation**: SDMetrics library, LibCity framework (optional)
- **Visualization**: matplotlib/plotly for trajectory plotting and analysis

---

## **Evaluation Strategy**

### **Multi-Faceted Validation Approach**

#### **1. Synthetic Data Quality**
> *"A multi-faceted evaluation approach, incorporating both quantitative metrics and qualitative assessments, is essential"* [[Anomalous_Taxi_Trajectory_Generation_.md#6-evaluation](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#6-evaluation-metrics)*

**SDMetrics Assessment:**
- **Resemblance**: Statistical similarity to real trajectories
- **Utility**: ML efficacy for downstream tasks
- **Privacy**: Disclosure protection analysis

#### **2. Anomaly Detection Performance**
**Key Metrics for Imbalanced Data:**
- **Precision**: Proportion of correctly identified anomalies
- **Recall**: Fraction of true anomalies successfully detected  
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **PR-AUC**: Area under precision-recall curve (better for imbalanced datasets) [[Anomalous_Taxi_Trajectory_Generation_.md#performance-metrics-for-anomaly-detection]](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md#performance-metrics-for-anomaly-detection)

#### **3. Cross-City Generalization**
- **Train on Beijing → Test on Chengdu/Xi'an**
- **Pattern Transfer Analysis**: Do Abp1-4 patterns generalize?
- **Performance Degradation**: Quantify cross-city performance drop

#### **4. Dataset Comparison Study**
- **Public vs. Private Beijing**: Performance comparison
- **Pattern Consistency**: Are anomaly patterns similar?
- **Privacy Analysis**: Synthetic data protection validation

---

## **Expected Deliverables & Impact**

### **Research Outputs**

#### **1. Multi-City Synthetic Anomaly Datasets**
- **4 Datasets**: Beijing (public/private), Chengdu, Xi'an, (Porto)
- **Labeled Anomalies**: Abp1-4 pattern classification
- **Quality Assured**: SDMetrics and LibCity validated

#### **2. Open-Source Framework**
**AnomalyTrajGen Components:**
- **Generation Pipeline**: DiffTraj + rule-based injection methods
- **Detection Baseline**: Isolation Forest + custom evaluation metrics
- **Evaluation Suite**: Cross-city and cross-dataset validation tools
- **Documentation**: Implementation guides and tutorials

#### **3. Research Contributions**
- **Cross-City Validation**: First comprehensive multi-city anomaly study
- **Benchmark Framework**: Standardized evaluation metrics
- **Public vs. Private Analysis**: Novel dataset comparison insights
- **Replication Package**: Complete reproducible research

### **Academic Impact**
- **Conference Paper**: DiffTraj-based anomaly generation with cross-city validation
- **Open Dataset**: Multi-city labeled synthetic anomalies for community
- **Benchmark**: Standard evaluation framework for future research

---

## **Risk Mitigation & Future Directions**

### **Risk Management**

#### **Technical Risks**
| Risk | Mitigation | Timeline Impact |
|------|------------|-----------------|
| **DiffTraj Setup Issues** | Use proven GitHub implementation | Low - Week 1 buffer |
| **Data Quality Problems** | Multiple datasets available | Low - Redundancy built-in |
| **Poor Anomaly Detection** | Focus on rule-based (controllable) | Medium - Adjust complexity |
| **Cross-City Poor Transfer** | Document negative results | Low - Still valuable research |

#### **Research Risks**
- **Limited Novelty**: Focus on **application** to **multi-city** + **cross-dataset** analysis
- **3-Month Constraint**: **Rule-based first**, advanced methods as future work
- **Dataset Access**: **Multiple options** reduce dependency risk

### **Future Research Directions**
> *Building on this foundation for advanced techniques:*

1. **Advanced Anomaly Injection**
   - Conditional generation with diffusion models
   - Latent space manipulation techniques
   - Physics-informed constraints
   
2. **Enhanced Detection Methods**
   - Transformer-based perplexity scoring
   - Reconstruction error optimization
   - User-specific anomaly modeling

3. **Privacy Enhancement**
   - Differential privacy integration
   - Formal privacy guarantees
   - Attack resistance validation

---


## **Success Criteria**
✅ **Technical**: Multi-dataset DiffTraj generation + rule-based anomaly injection  
✅ **Research**: Cross-city validation + public/private comparison  
✅ **Documentation**: Complete methodology + reproducible code  
✅ **Publication**: Conference-ready paper with novel contributions

---

## **References & Supporting Documentation**

### **Key Supporting Documents**
1. **[Anomalous_Taxi_Trajectory_Generation_.md](Meeting-Notes/Anomalous_Taxi_Trajectory_Generation_.md)** - Comprehensive research background
2. **[Methodology_Anomalous_Taxi_Trajectory_Generation_.md](Meeting-Notes/Methodology_Anomalous_Taxi_Trajectory_Generation_.md)** - Detailed technical methodology

### **External Resources**
- **DiffTraj Implementation**: [github.com/Yasoz/DiffTraj](https://github.com/Yasoz/DiffTraj)
- **T-Drive Dataset**: Microsoft Research Beijing taxi data
- **SDMetrics Library**: Synthetic data evaluation framework
- **LibCity Framework**: Urban trajectory analysis toolkit
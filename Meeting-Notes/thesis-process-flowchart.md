# Thesis Process Flowchart

## Overview
This diagram visualizes the complete research process as outlined in the main thesis document (main.tex), showing the flow from raw data through methodology implementation to final evaluation and results.

## Mermaid Diagram Code

```mermaid
graph TD
    A["Raw Beijing Taxi GPS Data<br/>(25.11.2019 - 01.12.2019)<br/>~16GB per day"] --> B["Data Quality Assessment"]
    
    B --> C["Data Preprocessing Pipeline"]
    C --> D["Data Cleaning & Error Correction"]
    D --> E["Trajectory Reconstruction"]
    E --> F["LibCity Format Standardization"]
    F --> G["Parameter Selection & Quality Control"]
    
    G --> H["Processed Dataset<br/>(LibCity Compatible)"]
    
    H --> I["Methodology 1:<br/>Abnormal Detection"]
    H --> J["Methodology 2:<br/>Trajectory Generation"]
    
    I --> K["Isolation Forest Analysis"]
    K --> L["Statistical Pattern Extraction"]
    L --> M["Enhanced Anomaly Framework"]
    M --> N["Indicators & Efficiency Analysis"]
    
    J --> O["LibCity Framework Extension"]
    O --> P["Statistical Model Architecture"]
    P --> Q["Graph-Based Generation"]
    Q --> R["Privacy Protection Mechanisms"]
    R --> S["Quality Assurance Framework"]
    
    N --> T["Real Data Analysis Results"]
    S --> U["Synthetic Dataset<br/>(LibCity Compatible)"]
    
    T --> V["Evaluation Phase"]
    U --> V
    
    V --> W["LibCity Benchmark Evaluation"]
    V --> X["Statistical Fidelity Assessment"]
    V --> Y["Privacy Preservation Assessment"]
    V --> Z["Computational Performance Analysis"]
    
    W --> AA["Cross-Model Comparison"]
    W --> BB["Multi-Task Performance"]
    
    X --> CC["Distribution Comparisons"]
    X --> DD["Anomaly Preservation"]
    
    Y --> EE["Attack Resistance Testing"]
    Y --> FF["Privacy-Utility Trade-off"]
    
    Z --> GG["Scalability Analysis"]
    
    AA --> HH["Final Results & Validation"]
    BB --> HH
    CC --> HH
    DD --> HH
    EE --> HH
    FF --> HH
    GG --> HH
    
    HH --> II["Research Contributions"]
    II --> JJ["Framework Extensions to LibCity"]
    II --> KK["Privacy-Preserving Synthetic Generation"]
    II --> LL["Cross-Dataset Compatibility"]
    
    classDef dataNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef methodNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef evalNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef resultNode fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A,B,H,U dataNode
    class I,J,K,L,M,N,O,P,Q,R,S methodNode
    class V,W,X,Y,Z,AA,BB,CC,DD,EE,FF,GG evalNode
    class T,HH,II,JJ,KK,LL resultNode
```

## Process Flow Description

### 1. Data Pipeline (Light Blue Nodes)
- **Raw Data**: Beijing taxi GPS data (25.11.2019 - 01.12.2019, ~16GB/day)
- **Quality Assessment**: Initial data validation and analysis
- **Processed Dataset**: LibCity-compatible standardized data format

### 2. Dual Methodology (Purple Nodes)
- **Methodology 1**: Isolation forest-based abnormal detection with pattern extraction
- **Methodology 2**: LibCity framework extension for synthetic trajectory generation

### 3. Evaluation Framework (Green Nodes)
- **LibCity Integration**: Leveraging existing benchmarks and evaluation tools
- **Multi-dimensional Assessment**: Statistical, privacy, and performance evaluation
- **Cross-validation**: Comparison with other models and datasets

### 4. Research Outcomes (Orange Nodes)
- **Framework Extensions**: Contributions to LibCity ecosystem
- **Privacy-Preserving Generation**: Novel synthetic data creation methods
- **Cross-Dataset Compatibility**: Standardized approach for broader research use

## Key Features

- **LibCity Integration**: Framework serves as backbone throughout entire process
- **Privacy by Design**: Protection mechanisms integrated into methodology, not added post-hoc
- **Comprehensive Evaluation**: Multi-faceted assessment using established benchmarks
- **Research Reproducibility**: Standardized formats enable comparison and replication
- **Extensibility**: Framework designed for use with multiple datasets and tasks

## File Information
- **Created**: Generated from main.tex thesis structure
- **Purpose**: Visual representation of complete research methodology
- **Usage**: Reference for understanding overall process flow and component relationships 
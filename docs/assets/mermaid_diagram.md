flowchart TD
    subgraph "Input Data Collection"
        A1["model_input_with_kinship_filtered_{dataset}.csv<br/>Genomic pair relationships<br/>Kinship labels: 1,2,3,4,5,6,UN"]
        A2["merged_info.out<br/>Primary statistical features<br/>103 columns"]
        A3["merged_added_info.out<br/>Additional statistical features<br/>123 columns with distance thresholds"]
    end

    subgraph "Data Integration & Processing"
        B1["Parse .out files<br/>Robust key:value extraction"]
        B2["Merge statistical data<br/>Handle column collisions"]
        B3["Integrate with genomic data<br/>Final dataset: 124+ columns"]
    end

    subgraph "Scenario Creation"
        C1["Included Scenario<br/>Keep all labels including UN<br/>~2,805 samples (cM_1/cM_3)<br/>~2,787 samples (cM_6)"]
        C2["NoUN Scenario<br/>Remove UN labels<br/>~882 samples for all variants<br/>Classes: 1,2,3,4,5,6 only"]
    end

    subgraph "Feature Engineering & Selection"
        D1["RandomForest Feature Selection<br/>Top 75 features"]
        D2["Aggregate Feature Creation<br/>Mathematical combinations"]
        D3["Feature Scaling<br/>Standardized transforms"]
    end

    subgraph "Model Training"
        E1["RandomForest<br/>Traditional ensemble"]
        E2["MLP<br/>5-layer neural network"]
        E3["CNN<br/>1D convolutional network"]
        
        F1["Zero Mode<br/>No rebalancing<br/>138.8-243.7s"]
        F2["Weighted Mode<br/>Class-weighted loss<br/>138.9-252.4s"]
        F3["SMOTE Mode<br/>Synthetic oversampling<br/>963.4-1,071.6s"]
        F4["OverUnder Mode<br/>SMOTE + ENN/Tomek<br/>197.1-244.7s"]
    end

    subgraph "Model Evaluation"
        G1["Performance Metrics<br/>F1, AUC, Accuracy"]
        G2["Confusion Matrices<br/>Detailed analysis"]
        G3["Model Comparisons<br/>24 combinations per variant"]
    end

    subgraph "Report Generation"
        H1["Performance Rankings<br/>RandomForest consistently best"]
        H2["Visualizations<br/>Plots and charts"]
        H3["Multi-format Output<br/>Markdown, PDF, bilingual"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C1
    B3 --> C2
    
    C1 --> D1
    C2 --> D1
    D1 --> D2
    D2 --> D3
    
    D3 --> E1
    D3 --> E2
    D3 --> E3
    
    E1 --> F1
    E1 --> F2
    E1 --> F3
    E1 --> F4
    E2 --> F1
    E2 --> F2
    E2 --> F3
    E2 --> F4
    E3 --> F1
    E3 --> F2
    E3 --> F3
    E3 --> F4
    
    F1 --> G1
    F2 --> G1
    F3 --> G1
    F4 --> G1
    G1 --> G2
    G2 --> G3
    G3 --> H1
    H1 --> H2
    H2 --> H3

    style A1 fill:#e1f5fe
    style A2 fill:#e1f5fe
    style A3 fill:#e1f5fe
    style C1 fill:#fff3e0
    style C2 fill:#fff3e0
    style G1 fill:#e8f5e8
    style H1 fill:#f3e5f5
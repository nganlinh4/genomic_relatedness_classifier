# Genomic Relatedness Classification Flow

This document describes the comprehensive step-by-step process of the genomic relatedness classification pipeline, from raw data preparation to final reporting. The pipeline handles three different centimorgan thresholds (cM_1, cM_3, cM_6) with similar flows but varying performance characteristics.

## Step 1: Raw Data Collection
- **Data files used**:
  - `model_input_with_kinship_filtered_{dataset}.csv` (primary labeled genomic data with kinship relationships)
  - `merged_info.out` (primary statistical features with 103 columns)
  - `merged_added_info.out` (additional statistical features with 123 columns including distance thresholds)
- **Process**: Load three separate data sources for each dataset variant
- **Results**: Raw data files loaded into memory for processing
- **Statistics**: The CSV contains genomic pair relationships with kinship labels (1, 2, 3, 4, 5, 6, UN), while the .out files contain statistical metrics from IBD analysis

## Step 2: Data Merging and Feature Integration
- **Data files used**: All three files from Step 1
- **Process**:
  - Parse statistical features from both .out files using robust key:value extraction
  - Merge primary and additional statistical data on 'pair' identifier
  - Handle column collisions by preferring primary values and backfilling with additional values
  - Merge statistical features with genomic labeled pairs data
- **Results**:
  - Single integrated dataset with kinship labels and statistical features
  - Final dataset contains all essential columns: pair, IBD1_len, IBD2_len, R1, R2, Num_Segs, Total_len, kinship
  - Statistical features include distance thresholds from <1cM to <10cM with raw counts and percentages
  - Output files: `data/processed/merged_<dataset>.csv` (UN-included scenario) and `data/processed/merged_<dataset>_noUN.csv` (UN-removed scenario)
- **Statistics**:
  - Primary statistical columns: 103 (excluding 'pair')
  - Additional statistical columns: 123 (excluding 'pair')
  - Common columns between primary and added: 104
  - Added-only columns: 20
  - Final merged dataset: 124 total columns plus kinship labels

## Step 3: Scenario Creation Based on Label Handling
- **Data files used**: Merged dataset from Step 2
- **Process**: Split into two scenarios based on handling of 'UN' (unknown) kinship labels
- **Results**:
  - **Included scenario**: Keeps all kinship labels including 'UN' (approximately 2,805 samples for cM_1, 2,805 samples for cM_3, 2,787 samples for cM_6)
  - **NoUN scenario**: Removes 'UN' labels and retains only specific kinship relationships (approximately 882 samples for all variants)
- **Statistics**:
  - Class distribution in included scenarios: 'UN' comprises 68-69% of all samples
  - Removed samples when creating NoUN scenario: approximately 1,923 samples
  - Kinship classes preserved: 1, 2, 3, 4, 5, 6 (unknown relationships excluded)

## Step 4: Feature Engineering and Selection
- **Data files used**: Scenario datasets from Step 3
- **Process**:
  - Apply statistical feature selection using RandomForest importance ranking
  - Identify top 75 most predictive features
  - Create additional aggregate features through mathematical combinations
  - Apply feature scaling using fitted scalers
- **Results**:
  - Optimized feature sets for each scenario
  - Trained scalers for consistent data transformation
  - Feature importance rankings and visualizations
  - Output files: Scaler files saved as `data/processed/scaler_<dataset><suffix>.pkl`, top features saved as `data/processed/top_features_<dataset><suffix>.pkl`, feature importance plots in `reports/<dataset>/assets/<scenario>/feature_importance_<dataset>_<scenario>.png` and `.svg`
- **Statistics**:
  - Selected features: Top 75 from available statistical variables
  - Feature types: IBD segments, distance thresholds, ratios, percentages
  - Scaler types: Standardized across all numerical features

## Step 5: Model Training with Class Imbalance Handling
- **Data files used**: Engineered and selected feature datasets from Step 4
- **Process**:
  - Train three different model architectures:
    - **RandomForest**: Traditional ensemble method with feature importance
    - **MLP**: Multi-layer perceptron neural network (5 layers)
    - **CNN**: 1D convolutional neural network for pattern recognition
  - Apply four class imbalance handling strategies:
    - **Zero**: No rebalancing (baseline, may favor majority class)
    - **Weighted**: Class-weighted loss functions during training
    - **SMOTE**: Synthetic minority oversampling technique
    - **OverUnder**: Combined SMOTE oversampling with ENN/Tomek under-sampling
- **Results**:
  - 24 trained models per dataset variant (2 scenarios × 4 modes × 3 models)
  - Model weights and configuration files
  - Training progress and performance logs
  - Output files: Trained neural network models saved as `models/<dataset>/<scenario>/<imbalance_mode>/<model>.pth` (MLP/CNN), training metadata in `models/<dataset>/<scenario>/<imbalance_mode>/training_meta.json`
- **Statistics**:
  - Training duration varies significantly:
    - Zero mode: 138.8-243.7 seconds
    - Weighted mode: 138.9-252.4 seconds
    - OverUnder mode: 197.1-244.7 seconds
    - SMOTE mode: 963.4-1,071.6 seconds (longest due to synthetic data generation)
  - All training performed on CUDA devices for efficiency
  - 100 training epochs per model

## Step 6: Comprehensive Model Evaluation
- **Data files used**: All trained models from Step 5, validation datasets
- **Process**:
  - Evaluate each model using multiple performance metrics
  - Generate confusion matrices for detailed classification analysis
  - Export probability predictions for further analysis
  - Compare performance across scenarios, modes, and model types
- **Results**:
  - Performance rankings for all 24 model combinations per variant
  - Detailed confusion matrices with visualizations
  - Probability predictions and confidence scores
  - Comparative analysis tables and charts
  - Output files: Performance results in `reports/<dataset>/results.json`, detailed reports in `reports/<dataset>/results.md`, confusion matrix plots in `reports/<dataset>/plots/confusion/<scenario>/<imbalance_mode>/`, probability predictions in `reports/<dataset>/assets/<scenario>/probabilities_<model>_<imbalance_mode>.json`
- **Statistics**:
  - **Best included scenario performance**:
    - cM_6: F1-weighted 0.9555, AUC-weighted 0.9962 (RandomForest overunder)
    - cM_1: F1-weighted 0.9460, AUC-weighted 0.9941 (RandomForest overunder)
    - cM_3: F1-weighted 0.9420, AUC-weighted 0.9928 (RandomForest overunder)
  - **Best noUN scenario performance**:
    - cM_1: F1-weighted 0.9234, AUC-weighted 0.9905 (RandomForest overunder)
    - cM_6: F1-weighted 0.9141, AUC-weighted 0.9874 (RandomForest overunder)
    - cM_3: F1-weighted 0.8929, AUC-weighted 0.9839 (RandomForest overunder)
  - **Consistent pattern**: RandomForest outperforms MLP and CNN across all variants
  - **Validation sample sizes**: ~697-702 (included), ~221 (noUN)

## Step 7: Automated Report Generation
- **Data files used**: Evaluation results from Step 6, model statistics, feature importance data
- **Process**:
  - Generate comprehensive markdown reports with executive summaries
  - Create visualizations including confusion matrix plots and performance charts
  - Produce bilingual reports (English and Korean) for accessibility
  - Generate PDF versions for sharing and archiving
  - Compile performance comparison tables across all variants
- **Results**:
  - Professional reports with detailed analysis and insights
  - Visual dashboards showing model performance comparisons
  - Feature importance plots and analysis
  - Training time and computational efficiency metrics
  - Output files: Bilingual reports in `reports/<dataset>/results.md` and `reports/<dataset>/results_KR.md`, visualization assets in `reports/<dataset>/assets/`, plots in `reports/<dataset>/plots/`, tables in `reports/<dataset>/tables/`, PDF versions
- **Statistics**:
  - Reports include detailed metrics: accuracy, F1 (weighted/macro), AUC (weighted/macro/micro)
  - Training time comparisons across different imbalance handling methods
  - Computational efficiency analysis and GPU utilization data
  - Comparative performance analysis across cM variants (cM_1, cM_3, cM_6)

## Variations Across cM Variants

### Performance Characteristics
- **cM_6**: Best overall performance in included scenarios, most computationally efficient
- **cM_1**: Best performance in noUN scenarios, balanced efficiency
- **cM_3**: Intermediate performance, longest training times for SMOTE mode

### Computational Patterns
- **Training efficiency ranking**: cM_6 > cM_1 > cM_3 (for most imbalance modes)
- **SMOTE mode**: Consistently requires 4-5x longer training time across all variants
- **GPU utilization**: All variants show efficient CUDA utilization during neural network training

### Data Structure Consistency
- All variants follow identical preprocessing steps and data structures
- Sample size variations are minimal (cM_6 has slightly fewer samples)
- Feature engineering process is standardized across all variants
- Model architectures and hyperparameters remain consistent

## Summary
The genomic relatedness classification pipeline represents a sophisticated machine learning system that successfully processes complex genomic data through multiple stages of feature engineering, model training, and evaluation. The system demonstrates robust performance across different centimorgan thresholds, with cM_6 showing the most promise for kinship classification tasks that include unknown relationships, while cM_1 provides the best results when unknown relationships are excluded from training.
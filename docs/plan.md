### Project Plan: Kinship Classification

The goal is to build and evaluate models that predict kinship. The features are a combination of 6 core IBD metrics plus a large set of distributional statistics parsed from `merged_info.out`. We create three separate datasets (`cM_1`, `cM_3`, `cM_6`) and generate a consolidated report per dataset (no cross-dataset comparison in a single report).

---

#### Step 1: Data Preparation (for each cM dataset)

1.  **Process `merged_info.out`:**
    *   Unzip and load the file (if needed).
    *   Parse each line as: `[pair] allChr <key1>:<val1> <key2>:<val2> ...` and build a row per pair with distributional features as columns (mean, std, percentiles, etc.).
    *   Produce a pandas DataFrame with one row per `pair` and columns for all parsed statistics (no transpose step).

2.  **Process `model_input_with_kinship_*.csv`:**
    *   Load the `model_input_with_kinship_filtered_<dataset>.csv` file.
    *   **Select only the essential columns:** `pair`, `IBD1_len`, `IBD2_len`, `R1`, `R2`, `Num_Segs`, `Total_len`, and the target column `kinship`.
    *   Discard all other columns from this file (`target1`, `target2`, `real_kinship`, `kinship_considered_hs`).

3.  **Merge Datasets:**
    *   Merge the parsed `merged_info` statistics with the selected columns from the `model_input` file on `pair`.
    *   We now branch into two scenarios to study the ambiguous `UN` label impact:
        - UN-included scenario: retain all rows (including `UN`).
        - UN-removed scenario: drop rows where `kinship == 'UN'`.
    *   **Deliverables:** `data/processed/merged_<dataset>.csv` (UN-included) and `data/processed/merged_<dataset>_noUN.csv` (UN-removed). Both are regenerable and ignored by Git.

#### Step 2: Exploratory Data Analysis (EDA)

1.  **Visualize Target Distribution:**
    *   Create a bar chart showing the count of each category in the `kinship` column.
    *   *Purpose:* Check for class imbalance.

#### Step 3: Feature Selection & Preprocessing

1.  **Define Features (X) and Target (y):**
    *   `X` = The combined features: the 6 core IBD features **plus** all distributional features from `merged_info`.
    *   `y` = The `kinship` column.

2.  **Scale the Features:**
    *   Apply `StandardScaler` to all feature (`X`) columns. This is essential.

3.  **Select Best Features:**
    *   Use a **`RandomForestClassifier`** to get feature importance scores from the combined feature set.
    *   Select the top-ranked features (top 50) to use for the deep learning models.
    *   Fit and persist a `StandardScaler` for the selected features.

#### Step 4: Model Building & Training (CUDA-only)

This is a multi-class classification problem. We train on GPU (CUDA required). We evaluate imbalance strategies per dataset and per scenario:

- `zero`: No class weighting, no resampling.
- `weighted`: Class-weighted loss (CrossEntropy with class weights).
- `smote`: Oversampling via SMOTE applied on the training split (no class weights to avoid double-compensation).
- `overunder`: Combined over + under sampling (SMOTE + ENN or SMOTE + Tomek) to both synthesize minority examples and remove ambiguous majority samples near decision boundaries.

1.  **Resampling Strategies:**
    *   `smote`: Apply SMOTE (Synthetic Minority Oversampling Technique) to the training set to generate synthetic samples for minority classes.
    *   `overunder`: Apply SMOTE then an under-sampling cleanup (e.g., ENN or Tomek links) to prune noisy/overlapping majority samples.
    *   `zero`: No resampling; raw class distributions.
    *   `weighted`: No resampling; rely on class-weighted loss.

2.  **Model A: Advanced Multi-Layer Perceptron (MLP)**
    *   **Architecture:** Deeper network with 4 hidden layers (256, 128, 64, 32 neurons), BatchNorm, Dropout (0.5), ReLU activations.
    *   **Input:** Scaled, selected features (oversampled only under `smote`).
    *   **Loss:** CrossEntropy; class-weighted only under `weighted`.

3.  **Model B: Advanced 1D Convolutional Neural Network (1D-CNN)**
    *   **Architecture:** Deeper CNN with 3 conv blocks (increasing filters: 32, 64, 128), each with Conv1d, BatchNorm, ReLU, MaxPool; followed by 2 dense layers (128, 64) with Dropout.
    *   **Input:** Scaled, selected features reshaped to 1D (oversampled only under `smote`).
    *   **Loss:** CrossEntropy; class-weighted only under `weighted`.

#### Step 5: Evaluation and Reporting

For both models, calculate and present the following metrics:

*   Accuracy (ACC)
*   F1-Score (weighted average)
*   AUC Score (One-vs-Rest): weighted, macro, micro — computed robustly with fallbacks (never N/A)
*   Confusion Matrix
*   Feature Importance Plot (from the Random Forest)

Notes on class imbalance:
- The 'zero' mode (no rebalancing) is a baseline and may be biased toward the majority class; prioritize macro/weighted metrics and per-class results.
- The 'weighted' mode applies class-weighted loss; the 'smote' mode oversamples the training split (validation remains original).
- The 'overunder' mode both oversamples minorities and removes borderline/ambiguous majority samples, often sharpening class boundaries.

Special training schedule:
- A `--special-epochs` override is applied only when the scenario includes UN and the strategy uses oversampling (`smote` or `overunder`). Other runs use the global `--epochs` value to avoid slowing the whole matrix.

#### Step 6: Repeat

*   Execute Steps 1-5 for all three datasets: `cM_1`, `cM_3`, and `cM_6`.
*   For each dataset, run both scenarios (UN-included and UN-removed) and all imbalance strategies (`zero`, `weighted`, `smote`, `overunder`).
*   For each dataset, a single consolidated report is written to `reports/<dataset>/` (English + Korean Markdown, optional PDFs). Both scenarios (`included` / `noUN`) are presented within the same report, with per-mode sections and confusion matrices. Scenario plots (kinship distribution, feature importance) are saved as both SVG and PNG and embedded with a two-column layout.

Research and follow-ups (stakeholder requests):
1. Compare results with and without oversampling.
2. Investigate UN label bias (scenario split + sampling strategies).
3. For non-oversampled runs, experiment with higher epochs to observe convergence differences.
4. Feature engineering: derive aggregate statistics (mean, std, min, max, quantiles) of top-importance features to test performance lift.
5. Tune `overunder` variant (choice of ENN vs Tomek) based on validation F1 and AUC robustness.

Note: A Korean version of this plan is available at `docs/plan_KR.md`.
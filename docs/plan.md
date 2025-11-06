### Project Plan: Kinship Classification

The goal is to build and evaluate models that predict kinship. The features will be a combination of 6 core IBD metrics and a large set of distributional stats. We will create three separate datasets (`cM_1`, `cM_3`, `cM_6`) and compare model performance on each.

---

#### Step 1: Data Preparation (for each cM dataset)

1.  **Process `merged_info.out`:**
    *   Unzip and load the file.
    *   **Transpose** it so that sample pairs are rows and distributional features (mean, std, percentiles, etc.) are columns.
    *   Convert it into a pandas DataFrame.

2.  **Process `model_input_with_kinship_*.csv`:**
    *   Load the `model_input_with_kinship_filtered_cM_1.csv` file.
    *   **Select only the essential columns:** `pair`, `IBD1_len`, `IBD2_len`, `R1`, `R2`, `Num_Segs`, `Total_len`, and the target column `kinship`.
    *   Discard all other columns from this file (`target1`, `target2`, `real_kinship`, `kinship_considered_hs`).

3.  **Merge Datasets:**
    *   Merge the processed `merged_info` DataFrame with the selected columns from the `model_input` DataFrame, using the `pair` column as the key.
    *   **Deliverable:** Share this final merged dataset.

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
    *   **Primary Method:** Use a **`RandomForestClassifier`** to get feature importance scores from the combined feature set.
    *   Select the top-ranked features to use for the deep learning models.

#### Step 4: Model Building & Training

*This remains a multi-class classification problem. Due to severe class imbalance, apply SMOTE oversampling to balance training data, and use advanced model architectures for better performance.*

1.  **Data Oversampling:**
    *   Apply SMOTE (Synthetic Minority Oversampling Technique) to the training set to generate synthetic samples for minority classes, balancing the dataset.

2.  **Model A: Advanced Multi-Layer Perceptron (MLP)**
    *   **Architecture:** Deeper network with 4 hidden layers (256, 128, 64, 32 neurons), BatchNorm, Dropout (0.5), ReLU activations.
    *   **Input:** Oversampled, scaled, selected features.
    *   **Output:** Dense layer with softmax and weighted categorical crossentropy loss.

3.  **Model B: Advanced 1D Convolutional Neural Network (1D-CNN)**
    *   **Architecture:** Deeper CNN with 3 conv blocks (increasing filters: 32, 64, 128), each with Conv1d, BatchNorm, ReLU, MaxPool; followed by 2 dense layers (128, 64) with Dropout.
    *   **Input:** Oversampled, scaled, selected features.
    *   **Output:** Dense layer with softmax and weighted categorical crossentropy loss.

#### Step 5: Evaluation and Reporting

For both models, calculate and present the following metrics:

*   Accuracy (ACC)
*   F1-Score (weighted average)
*   AUC Score (One-vs-Rest)
*   Confusion Matrix
*   Feature Importance Plot (from the Random Forest)

#### Step 6: Repeat

*   Execute Steps 1-5 for all three datasets: `cM_1`, `cM_3`, and `cM_6`.
*   The final report will compare the results across the three datasets.
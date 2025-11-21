# Rebalancing Methods in Genomic Relatedness Classifier

This document provides a comprehensive explanation of the four class rebalancing strategies implemented in the `genomic_relatedness_classifier` project. These methods are designed to address class imbalance issues in the training data, ensuring that the models (MLP, CNN, Random Forest) can effectively learn to classify kinship relationships even when certain classes are underrepresented.

The implementation logic discussed below is primarily located in `scripts/train_models.py`.

To illustrate the impact of each method, we use the **cM_1 dataset** as a case study.

## Case Study Data: cM_1
The original training distribution of the `cM_1` dataset is highly imbalanced, dominated by the "UN" (Unrelated) class.

**Original Class Counts (Training Set Approximation):**
*   **UN**: ~2,404 samples (Majority)
*   **4th Degree**: ~338 samples
*   **5th Degree**: ~266 samples
*   **3rd Degree**: ~184 samples
*   **2nd Degree**: ~144 samples
*   **1st Degree**: ~127 samples
*   **6th Degree**: ~44 samples (Minority)

*Total Samples: ~3,507*

---

## 1. Zero (Baseline)

### What
The "Zero" method (referred to as `zero` in the CLI arguments) represents the baseline training approach where **no explicit rebalancing** is performed. The model is trained on the dataset exactly as it is provided.

### How
*   **Data**: The model sees the raw distribution: 2,404 "UN" samples vs only 44 "6th Degree" samples.
*   **Loss Function**: Standard `nn.CrossEntropyLoss()` with no weights.

### Why
*   **Baseline Comparison**: It serves as a control.
*   **Performance**: In our tests (MLP on cM_1), this achieved an **F1-Macro score of 0.4737**. The model likely biases heavily towards the "UN" class, achieving high accuracy but failing to correctly identify closer relationships.

---

## 2. Weighted (Class Weighting)

### What
The "Weighted" method (`weighted`) addresses imbalance algorithmically within the loss function. It assigns a higher penalty for misclassifying minority classes.

### How
*   **Weight Calculation**: Weights are inversely proportional to class frequency.
    *   Weight for "6th Degree" (44 samples) is roughly **54x higher** than the weight for "UN" (2404 samples).
*   **Loss Function**: `nn.CrossEntropyLoss(weight=class_weights)`.

### Why
*   **Cost-Sensitive Learning**: Forces the model to pay attention to the minority.
*   **Performance**: Achieved an **F1-Macro score of 0.4363**.
    *   *Observation*: Surprisingly, this performed worse than baseline in this specific case. This can happen if the weights are too aggressive, causing the model to over-focus on noisy minority examples or destabilizing the gradient descent.

---

## 3. SMOTE (Synthetic Minority Over-sampling Technique)

### What
SMOTE (`smote`) creates **synthetic** new examples by interpolating between existing minority samples to match the majority class count.

### How
*   **Target**: All classes are oversampled to match the majority class count (UN: 2,404).
*   **Resulting Distribution**:
    *   UN: 2,404 (Original)
    *   6th Degree: 44 Original + ~2,360 Synthetic = 2,404
    *   ...and so on for all classes.
*   **Total Training Size**: 2,404 * 7 classes = **16,828 samples**.
*   **Configuration**: Uses `k_neighbors=1` to handle extremely small classes.

### Why
*   **Full Representation**: Ensures the model sees an equal number of examples for every class.
*   **Performance**: Achieved an **F1-Macro score of 0.5207**. This is a significant improvement over the baseline, proving that generating synthetic data helps the model generalize better on minority classes.

---

## 4. OverUnder (Hybrid Sampling)

### What
The "OverUnder" method (`overunder`) is a sophisticated hybrid strategy that combines **oversampling**, **undersampling**, and **data cleaning** to reach a specific, efficient target size.

### How
1.  **Target Definition**: Fixed target of **400 samples per class**.
2.  **Oversampling (SMOTE)**: Minority classes (e.g., 6th Degree with 44) are SMOTE-ed up to 400.
3.  **Undersampling**: The majority class (UN with 2,404) is randomly downsampled to 400.
4.  **Cleaning (ENN/Tomek)**: Algorithms like SMOTEENN remove "noisy" or borderline samples to clarify decision boundaries.
5.  **Resulting Distribution**: ~400 samples per class.
*   **Total Training Size**: ~2,800 samples.

### Why
*   **Efficiency**: It reduces the dataset size from ~16,828 (SMOTE) to ~2,800, making training **6x faster** while maintaining balance.
*   **Quality**: The cleaning step removes ambiguous samples.
*   **Performance**: Achieved the **highest F1-Macro score of 0.5372**.
    *   *Conclusion*: This method provides the best balance of accuracy and computational efficiency, outperforming both the baseline and pure SMOTE.

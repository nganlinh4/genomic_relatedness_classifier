# Results for cM_1

Generated: 2025-11-06T05:07:35.386866Z  
Device: cuda

Labels: 1, 2, 3, 4, 5, 6, UN

## Mode: zero

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.9117 | 0.8696 | 0.1590 | N/A | N/A | N/A |
| CNN | 0.9117 | 0.8696 | 0.1590 | N/A | N/A | N/A |
| RandomForest | 0.8789 | 0.8551 | 0.1929 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](zero\confusion_matrix_mlp_cM_1_zero.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](zero\confusion_matrix_cnn_cM_1_zero.png)

</details>

## Mode: weighted

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.0627 | 0.0099 | 0.0847 | N/A | N/A | N/A |
| CNN | 0.0926 | 0.0668 | 0.1204 | N/A | N/A | N/A |
| RandomForest | 0.8789 | 0.8551 | 0.1929 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](weighted\confusion_matrix_mlp_cM_1_weighted.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](weighted\confusion_matrix_cnn_cM_1_weighted.png)

</details>

## Mode: smote

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.0427 | 0.0353 | 0.1352 | N/A | N/A | N/A |
| CNN | 0.5798 | 0.6812 | 0.2658 | N/A | N/A | N/A |
| RandomForest | 0.8789 | 0.8551 | 0.1929 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](smote\confusion_matrix_mlp_cM_1_smote.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](smote\confusion_matrix_cnn_cM_1_smote.png)

</details>

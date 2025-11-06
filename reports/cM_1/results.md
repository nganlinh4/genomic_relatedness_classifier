# Results for cM_1

Generated: 2025-11-06T06:02:36.420580Z  
Device: cuda

Labels: 1, 2, 3, 4, 5, 6, UN

## Mode: zero

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.9088 | 0.8681 | 0.1587 | 0.7264 | 0.7350 | 0.9714 |
| CNN | 0.9117 | 0.8696 | 0.1590 | 0.6842 | 0.7595 | 0.9834 |
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
| MLP | 0.1880 | 0.2316 | 0.1375 | 0.7468 | 0.7770 | 0.8457 |
| CNN | 0.0655 | 0.0169 | 0.1073 | 0.6621 | 0.7494 | 0.8281 |
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
| MLP | 0.0427 | 0.0338 | 0.1342 | 0.6994 | 0.7987 | 0.7729 |
| CNN | 0.6168 | 0.7145 | 0.2804 | 0.8293 | 0.8372 | 0.8981 |
| RandomForest | 0.8789 | 0.8551 | 0.1929 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](smote\confusion_matrix_mlp_cM_1_smote.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](smote\confusion_matrix_cnn_cM_1_smote.png)

</details>

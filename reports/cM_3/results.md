# Results for cM_3

Generated: 2025-11-06T05:09:50.525978Z  
Device: cpu

Labels: 1, 2, 3, 4, 5, 6, UN

## Mode: zero

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.9117 | 0.8696 | 0.1590 | N/A | N/A | N/A |
| CNN | 0.9117 | 0.8696 | 0.1590 | N/A | N/A | N/A |
| RandomForest | 0.8761 | 0.8534 | 0.1890 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](zero\confusion_matrix_mlp_cM_3_zero.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](zero\confusion_matrix_cnn_cM_3_zero.png)

</details>

## Mode: weighted

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.8675 | 0.8515 | 0.2302 | N/A | N/A | N/A |
| CNN | 0.1709 | 0.1942 | 0.1305 | N/A | N/A | N/A |
| RandomForest | 0.8761 | 0.8534 | 0.1890 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](weighted\confusion_matrix_mlp_cM_3_weighted.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](weighted\confusion_matrix_cnn_cM_3_weighted.png)

</details>

## Mode: smote

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.0342 | 0.0237 | 0.1234 | N/A | N/A | N/A |
| CNN | 0.3718 | 0.5028 | 0.2146 | N/A | N/A | N/A |
| RandomForest | 0.8761 | 0.8534 | 0.1890 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](smote\confusion_matrix_mlp_cM_3_smote.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](smote\confusion_matrix_cnn_cM_3_smote.png)

</details>

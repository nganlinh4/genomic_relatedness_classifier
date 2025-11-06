# Results for cM_3

Generated: 2025-11-06T06:03:39.139866Z  
Device: cuda

Labels: 1, 2, 3, 4, 5, 6, UN

## Mode: zero

Validation samples: 702  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=640

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.9117 | 0.8696 | 0.1590 | 0.7031 | 0.6960 | 0.9692 |
| CNN | 0.9046 | 0.8660 | 0.1583 | 0.6980 | 0.7703 | 0.9826 |
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
| MLP | 0.0741 | 0.0363 | 0.1110 | 0.5057 | 0.6411 | 0.8137 |
| CNN | 0.0698 | 0.0127 | 0.1188 | 0.7578 | 0.7715 | 0.7450 |
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
| MLP | 0.0470 | 0.0404 | 0.1346 | 0.7725 | 0.8255 | 0.7562 |
| CNN | 0.3932 | 0.5221 | 0.2216 | 0.7414 | 0.8231 | 0.7869 |
| RandomForest | 0.8761 | 0.8534 | 0.1890 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](smote\confusion_matrix_mlp_cM_3_smote.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](smote\confusion_matrix_cnn_cM_3_smote.png)

</details>

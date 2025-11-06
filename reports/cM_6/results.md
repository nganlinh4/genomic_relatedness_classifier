# Results for cM_6

Generated: 2025-11-06T06:04:30.385268Z  
Device: cuda

Labels: 1, 2, 3, 4, 5, 6, UN

## Mode: zero

Validation samples: 697  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=635

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.9082 | 0.8672 | 0.1586 | 0.7576 | 0.6892 | 0.9743 |
| CNN | 0.9110 | 0.8686 | 0.1589 | 0.7025 | 0.7732 | 0.9851 |
| RandomForest | 0.8809 | 0.8572 | 0.2227 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](zero\confusion_matrix_mlp_cM_6_zero.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](zero\confusion_matrix_cnn_cM_6_zero.png)

</details>

## Mode: weighted

Validation samples: 697  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=635

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.0617 | 0.0102 | 0.0902 | 0.6369 | 0.7134 | 0.6999 |
| CNN | 0.6370 | 0.7213 | 0.2559 | 0.7847 | 0.8150 | 0.9239 |
| RandomForest | 0.8809 | 0.8572 | 0.2227 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](weighted\confusion_matrix_mlp_cM_6_weighted.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](weighted\confusion_matrix_cnn_cM_6_weighted.png)

</details>

## Mode: smote

Validation samples: 697  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=635

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.0287 | 0.0145 | 0.1161 | 0.7599 | 0.8075 | 0.7525 |
| CNN | 0.6628 | 0.7456 | 0.2990 | 0.8204 | 0.8213 | 0.9285 |
| RandomForest | 0.8809 | 0.8572 | 0.2227 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](smote\confusion_matrix_mlp_cM_6_smote.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](smote\confusion_matrix_cnn_cM_6_smote.png)

</details>

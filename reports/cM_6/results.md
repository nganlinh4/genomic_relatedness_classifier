# Results for cM_6

Generated: 2025-11-06T05:11:22.024952Z  
Device: cpu

Labels: 1, 2, 3, 4, 5, 6, UN

## Mode: zero

Validation samples: 697  
Class distribution: 1=7, 2=7, 3=9, 4=38, 5=1, 6=0, UN=635

| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |
|-------|----------|---------------|------------|----------------|-------------|-------------|
| MLP | 0.9110 | 0.8686 | 0.1589 | N/A | N/A | N/A |
| CNN | 0.9110 | 0.8686 | 0.1589 | N/A | N/A | N/A |
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
| MLP | 0.0617 | 0.0081 | 0.0530 | N/A | N/A | N/A |
| CNN | 0.3228 | 0.4131 | 0.2218 | N/A | N/A | N/A |
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
| MLP | 0.0330 | 0.0207 | 0.1240 | N/A | N/A | N/A |
| CNN | 0.4849 | 0.6000 | 0.2440 | N/A | N/A | N/A |
| RandomForest | 0.8809 | 0.8572 | 0.2227 | N/A | N/A | N/A |

<details><summary>Confusion Matrix: MLP</summary>

![](smote\confusion_matrix_mlp_cM_6_smote.png)

</details>
<details><summary>Confusion Matrix: CNN</summary>

![](smote\confusion_matrix_cnn_cM_6_smote.png)

</details>

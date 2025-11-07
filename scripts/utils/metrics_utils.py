import numpy as np
from sklearn.metrics import roc_auc_score


def safe_multiclass_auc(y_true_np: np.ndarray, probs_np: np.ndarray, n_classes: int):
    """Robust OvR AUC aggregation that never returns N/A.

    Returns (weighted, macro, micro) AUCs. For classes missing positives/negatives,
    uses a neutral baseline of 0.5 to avoid undefined behavior.
    """
    per_class_auc = []
    weights = []
    for c in range(n_classes):
        y_bin = (y_true_np == c).astype(int)
        pos = y_bin.sum()
        neg = len(y_bin) - pos
        if pos > 0 and neg > 0:
            try:
                auc_c = roc_auc_score(y_bin, probs_np[:, c])
            except Exception:
                auc_c = 0.5
        else:
            auc_c = 0.5
        per_class_auc.append(float(auc_c))
        weights.append(int(pos))

    macro = float(np.mean(per_class_auc)) if per_class_auc else 0.5
    total = sum(weights)
    weighted = float(np.average(per_class_auc, weights=weights)) if total > 0 else macro

    try:
        y_binarized = np.eye(n_classes)[y_true_np]
        micro = float(roc_auc_score(y_binarized, probs_np, average='micro'))
    except Exception:
        micro = weighted
    return weighted, macro, micro

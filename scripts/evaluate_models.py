import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import json
import argparse

# CLI
parser = argparse.ArgumentParser(description='Evaluate models for a dataset and imbalance mode')
parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6')
parser.add_argument('imbalance_mode', type=str, choices=['zero','weighted','smote'], help='imbalance handling mode')
parser.add_argument('--eval-device', type=str, choices=['cpu','cuda'], default=None, help='Evaluation device (default cuda; required to be available)')
args = parser.parse_args()

dataset = args.dataset
imbalance_mode = args.imbalance_mode

# Load data
df = pd.read_csv(f'data/processed/merged_{dataset}.csv')

# Load top features and scaler
with open(f'data/processed/top_features_{dataset}.pkl', 'rb') as f:
    top_features = pickle.load(f)

with open(f'data/processed/scaler_{dataset}.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare features and target
X = df[top_features]
y = df['kinship']

# Encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
X_scaled = scaler.transform(X)

# Split into train/val (same as training, but since no test, evaluate on val)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Device selection: enforce CUDA by default; error if not available when requested
eval_device_env = (args.eval_device or os.environ.get('EVAL_DEVICE', 'cuda')).lower()
if eval_device_env == 'cuda':
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for evaluation but is not available. Please ensure CUDA is available.")
    device = torch.device('cuda')
elif eval_device_env == 'cpu':
    raise SystemExit("CPU evaluation is disabled per project policy. Use --eval-device cuda and ensure CUDA is available.")
else:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by default for evaluation but not available.")
    device = torch.device('cuda')
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# Define models (advanced versions)
class AdvancedMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AdvancedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class AdvancedCNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AdvancedCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (input_size // 8), 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

input_size = len(top_features)
num_classes = len(le.classes_)

# Load models
mlp = AdvancedMLP(input_size, num_classes).to(device)
mlp_path = os.path.join('models', dataset, imbalance_mode, 'mlp.pth')
mlp.load_state_dict(torch.load(mlp_path, map_location=device))
mlp.eval()

cnn = AdvancedCNN1D(input_size, num_classes).to(device)
cnn_path = os.path.join('models', dataset, imbalance_mode, 'cnn.pth')
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

# Evaluate function
def _safe_multiclass_auc(y_true_np: np.ndarray, probs_np: np.ndarray, n_classes: int):
    # Compute per-class OvR AUC; use 0.5 baseline for classes absent in y_true or degenerate cases
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
            # If no positives or no negatives, AUC undefined; use neutral baseline 0.5
            auc_c = 0.5
        per_class_auc.append(float(auc_c))
        weights.append(int(pos))

    macro = float(np.mean(per_class_auc)) if per_class_auc else 0.5
    total = sum(weights)
    weighted = float(np.average(per_class_auc, weights=weights)) if total > 0 else macro

    # Micro AUC may fail if degenerate; fall back to weighted
    try:
        y_binarized = np.eye(n_classes)[y_true_np]
        micro = float(roc_auc_score(y_binarized, probs_np, average='micro'))
    except Exception:
        micro = weighted
    return weighted, macro, micro


def evaluate_model(model, X, y, model_key, pretty_name):
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        y_true = y.cpu().numpy()

    acc = accuracy_score(y_true, predicted)
    f1_weighted = f1_score(y_true, predicted, average='weighted')
    f1_macro = f1_score(y_true, predicted, average='macro')

    # AUC (One-vs-Rest) with robust fallbacks to avoid N/A
    auc_weighted, auc_macro, auc_micro = _safe_multiclass_auc(y_true, probs, num_classes)

    cm = confusion_matrix(y_true, predicted)

    print(f"\n{pretty_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    print(f"AUC (OvR, weighted): {auc_weighted:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    unique_labels = np.unique(np.concatenate([y_true, predicted]))
    target_names = [le.classes_[i] for i in unique_labels]
    cls_report = classification_report(y_true, predicted, labels=unique_labels, target_names=target_names, output_dict=True)
    print(classification_report(y_true, predicted, labels=unique_labels, target_names=target_names))

    # Plot confusion matrix
    out_dir = os.path.join('reports', dataset, imbalance_mode)
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, f'confusion_matrix_{model_key}_{dataset}_{imbalance_mode}.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {pretty_name} ({dataset} / {imbalance_mode})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'auc_weighted': auc_weighted,
        'auc_macro': auc_macro,
        'auc_micro': auc_micro,
        'confusion_matrix_path': cm_path,
        'per_class': cls_report
    }

# Evaluate DL models
mlp_metrics = evaluate_model(mlp, X_val_tensor, y_val_tensor, 'mlp', f"Advanced MLP")
cnn_metrics = evaluate_model(cnn, X_val_tensor, y_val_tensor, 'cnn', f"Advanced 1D-CNN")

# Evaluate Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

acc_rf = accuracy_score(y_val, y_pred_rf)
f1_rf = f1_score(y_val, y_pred_rf, average='weighted')

print("\nRandom Forest Baseline Results:")
print(f"Accuracy: {acc_rf:.4f}")
print(f"F1-Score (weighted): {f1_rf:.4f}")
print("Classification Report:")
unique_labels_rf = np.unique(np.concatenate([y_val, y_pred_rf]))
target_names_rf = [le.classes_[i] for i in unique_labels_rf]
print(classification_report(y_val, y_pred_rf, labels=unique_labels_rf, target_names=target_names_rf))

rf_metrics = {
    'accuracy': acc_rf,
    'f1_weighted': f1_rf,
    'f1_macro': f1_score(y_val, y_pred_rf, average='macro'),
    'auc_weighted': None,
    'auc_macro': None,
    'auc_micro': None,
    'confusion_matrix_path': None,
    'per_class': classification_report(y_val, y_pred_rf, output_dict=True)
}

# Save metrics to JSON
results = {
    'dataset': dataset,
    'imbalance_mode': imbalance_mode,
    'device': str(device),
    'label_names': list(le.classes_),
    'val_samples': int(len(y_val)),
    'val_class_distribution': {le.classes_[i]: int((y_val == i).sum()) for i in range(len(le.classes_))},
    'models': {
        'MLP': mlp_metrics,
        'CNN': cnn_metrics,
        'RandomForest': rf_metrics
    }
}

with open(f'data/processed/evaluation_results_{dataset}_{imbalance_mode}.json', 'w') as f:
    json.dump(results, f, indent=2)

# Feature importance plot (already done in feature_selection, but reload and show)
print(f"\nFeature importance plot generated in reports/{dataset}/feature_importance_{dataset}.png if feature_selection was run.")
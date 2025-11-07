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

# Ensure repository root is on sys.path when script executed directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from imblearn.over_sampling import SMOTE
try:
    from imblearn.combine import SMOTEENN, SMOTETomek
except ImportError:
    SMOTEENN = None
    SMOTETomek = None

# Add parent directory to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from scripts.utils.metrics_utils import safe_multiclass_auc
except ModuleNotFoundError:
    # Fallback: attempt relative import if working directory changed
    try:
        from utils.metrics_utils import safe_multiclass_auc  # type: ignore
    except ModuleNotFoundError:
        raise

# CLI
parser = argparse.ArgumentParser(description='Evaluate models for a dataset, scenario, and imbalance mode')
parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6')
parser.add_argument('imbalance_mode', type=str, choices=['zero','weighted','smote','overunder'], help='imbalance handling mode')
parser.add_argument('--scenario', type=str, choices=['included','noUN'], default='included', help='Scenario: included (retain UN) or noUN (drop UN)')
parser.add_argument('--eval-device', type=str, choices=['cpu','cuda'], default=None, help='Evaluation device (default cuda; required to be available)')
args = parser.parse_args()

dataset = args.dataset
scenario = args.scenario
imbalance_mode = args.imbalance_mode

# Load data
suffix = '' if scenario == 'included' else '_noUN'
df = pd.read_csv(f'data/processed/merged_{dataset}{suffix}.csv')

# Load top features and scaler
with open(f'data/processed/top_features_{dataset}{suffix}.pkl', 'rb') as f:
    top_features = pickle.load(f)

with open(f'data/processed/scaler_{dataset}{suffix}.pkl', 'rb') as f:
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
mlp_path = os.path.join('models', dataset, scenario, imbalance_mode, 'mlp.pth')
mlp.load_state_dict(torch.load(mlp_path, map_location=device))
mlp.eval()

cnn = AdvancedCNN1D(input_size, num_classes).to(device)
cnn_path = os.path.join('models', dataset, scenario, imbalance_mode, 'cnn.pth')
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

# Evaluate function
def _safe_multiclass_auc(y_true_np: np.ndarray, probs_np: np.ndarray, n_classes: int):
    return safe_multiclass_auc(y_true_np, probs_np, n_classes)


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
    target_names = [str(le.classes_[i]) for i in unique_labels]
    cls_report = classification_report(y_true, predicted, labels=unique_labels, target_names=target_names, output_dict=True)
    print(classification_report(y_true, predicted, labels=unique_labels, target_names=target_names))

    # Plot confusion matrix into organized directory
    out_dir = os.path.join('reports', dataset, 'plots', 'confusion', scenario, imbalance_mode)
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, f'confusion_matrix_{model_key}_{dataset}_{scenario}_{imbalance_mode}.png')
    plt.figure(figsize=(12, 10), dpi=200)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cbar=False,
        annot_kws={"size": 12}
    )
    plt.title(f'Confusion Matrix - {pretty_name} ({dataset} / {scenario} / {imbalance_mode})', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
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

# Accuracy and F1
acc_rf = accuracy_score(y_val, y_pred_rf)
f1_rf_weighted = f1_score(y_val, y_pred_rf, average='weighted')
f1_rf_macro = f1_score(y_val, y_pred_rf, average='macro')

# AUC using predict_proba (robustly fill missing class columns)
try:
    proba_partial = rf.predict_proba(X_val)  # shape: [n, len(rf.classes_)]
    rf_classes = rf.classes_.astype(int)
    probs_full = np.zeros((X_val.shape[0], num_classes), dtype=float)
    for idx_c, c in enumerate(rf_classes):
        probs_full[:, c] = proba_partial[:, idx_c]
    auc_w_rf, auc_m_rf, auc_micro_rf = _safe_multiclass_auc(y_val, probs_full, num_classes)
except Exception:
    auc_w_rf = auc_m_rf = auc_micro_rf = None

# Confusion matrix and report
cm_rf = confusion_matrix(y_val, y_pred_rf)
print("\nRandom Forest Baseline Results:")
print(f"Accuracy: {acc_rf:.4f}")
print(f"F1-Score (weighted): {f1_rf_weighted:.4f}")
if auc_w_rf is not None:
    print(f"AUC (OvR, weighted): {auc_w_rf:.4f}")
print("Classification Report:")
unique_labels_rf = np.unique(np.concatenate([y_val, y_pred_rf]))
target_names_rf = [str(le.classes_[i]) for i in unique_labels_rf]
print(classification_report(y_val, y_pred_rf, labels=unique_labels_rf, target_names=target_names_rf))

# Plot confusion matrix for RF
out_dir_mode = os.path.join('reports', dataset, 'plots', 'confusion', scenario, imbalance_mode)
os.makedirs(out_dir_mode, exist_ok=True)
cm_rf_path = os.path.join(out_dir_mode, f'confusion_matrix_rf_{dataset}_{scenario}_{imbalance_mode}.png')
plt.figure(figsize=(12, 10), dpi=200)
sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cbar=False,
    annot_kws={"size": 12}
)
plt.title(f'Confusion Matrix - RandomForest ({dataset} / {scenario} / {imbalance_mode})', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(cm_rf_path)
plt.close()

rf_metrics = {
    'accuracy': acc_rf,
    'f1_weighted': f1_rf_weighted,
    'f1_macro': f1_rf_macro,
    'auc_weighted': auc_w_rf,
    'auc_macro': auc_m_rf,
    'auc_micro': auc_micro_rf,
    'confusion_matrix_path': cm_rf_path,
    'per_class': classification_report(y_val, y_pred_rf, output_dict=True)
}

# Save metrics to JSON
results = {
    'dataset': dataset,
    'scenario': scenario,
    'imbalance_mode': imbalance_mode,
    'device': str(device),
    'label_names': list(le.classes_),
    'val_samples': int(len(y_val)),
    'val_class_distribution': {le.classes_[i]: int((y_val == i).sum()) for i in range(len(le.classes_))},
    'train_samples_before': int(len(y_train)),
    'train_class_distribution_before': {le.classes_[i]: int((y_train == i).sum()) for i in range(len(le.classes_))},
    'train_samples_after': None,
    'train_class_distribution_after': None,
    'models': {
        'MLP': mlp_metrics,
        'CNN': cnn_metrics,
        'RandomForest': rf_metrics
    }
}

# Compute post-sampling train counts for reporting (mirrors training policy)
try:
    if imbalance_mode in ['smote', 'overunder']:
        class_counts = np.bincount(y_train, minlength=num_classes)
        max_count = int(class_counts.max())
        if imbalance_mode == 'smote':
            target_per_class = max_count
            sampling_strategy = {cls_idx: target_per_class for cls_idx in range(num_classes)}
        else:
            target_per_class = 300
            sampling_strategy = {cls_idx: target_per_class for cls_idx in range(num_classes) if int(class_counts[cls_idx]) < target_per_class}
        if sampling_strategy:
            sm = SMOTE(k_neighbors=1, random_state=42, sampling_strategy=sampling_strategy)
            X_res, y_res = sm.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
        if imbalance_mode == 'overunder':
            # Optional cleaning mirror (no effect on final counts we report)
            if SMOTEENN is not None:
                try:
                    se = SMOTEENN(random_state=42)
                    X_res, y_res = se.fit_resample(X_res, y_res)
                except Exception:
                    pass
            elif SMOTETomek is not None:
                try:
                    st = SMOTETomek(random_state=42)
                    X_res, y_res = st.fit_resample(X_res, y_res)
                except Exception:
                    pass
            # Clip to exactly 300 if any cleaning overshoots
            final_indices = []
            for cls_idx in range(num_classes):
                cls_mask = np.where(y_res == cls_idx)[0]
                if len(cls_mask) > target_per_class:
                    cls_mask = np.random.default_rng(42).choice(cls_mask, size=target_per_class, replace=False)
                final_indices.extend(cls_mask.tolist())
            final_indices = np.array(final_indices)
            X_res = X_res[final_indices]
            y_res = y_res[final_indices]
        y_train_res = y_res
        results['train_samples_after'] = int(len(y_train_res))
        results['train_class_distribution_after'] = {le.classes_[i]: int((y_train_res == i).sum()) for i in range(len(le.classes_))}
    else:
        results['train_samples_after'] = int(len(y_train))
        results['train_class_distribution_after'] = results['train_class_distribution_before']
except Exception:
    # If SMOTE or counting fails, fall back gracefully
    results['train_samples_after'] = int(len(y_train))
    results['train_class_distribution_after'] = results['train_class_distribution_before']

def convert_numpy_types(obj):
    """Recursively convert numpy types to built-ins and ensure dict keys are JSON-compatible."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Normalize key types (numpy scalars -> str of Python value to avoid TypeError)
            if isinstance(key, (np.integer, np.floating)):
                norm_key = str(key.item())
            else:
                norm_key = key if isinstance(key, (str, int, float, bool, type(None))) else str(key)
            new_dict[norm_key] = convert_numpy_types(value)
        return new_dict
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

with open(f'data/processed/evaluation_results_{dataset}_{scenario}_{imbalance_mode}.json', 'w') as f:
    json.dump(convert_numpy_types(results), f, indent=2)

# Feature importance plot (already done in feature_selection, but reload and show)
assets_dir = os.path.join('reports', dataset, 'assets', scenario)
print(f"\nFeature importance plot generated in {os.path.join(assets_dir, f'feature_importance_{dataset}_{scenario}.png')} if feature_selection was run.")
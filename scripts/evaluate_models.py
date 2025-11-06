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
import json

# Check arguments
if len(sys.argv) < 3:
    print("Usage: python evaluate_models.py <dataset> <imbalance_mode>")
    print("dataset: cM_1, cM_3, cM_6")
    print("imbalance_mode: zero, weighted, smote")
    sys.exit(1)

dataset = sys.argv[1]
imbalance_mode = sys.argv[2]

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

# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
mlp.load_state_dict(torch.load(f'data/processed/mlp_{dataset}_{imbalance_mode}.pth', weights_only=True))
mlp.eval()

cnn = AdvancedCNN1D(input_size, num_classes).to(device)
cnn.load_state_dict(torch.load(f'data/processed/cnn_{dataset}_{imbalance_mode}.pth', weights_only=True))
cnn.eval()

# Evaluate function
def evaluate_model(model, X, y, model_name):
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        y_true = y.cpu().numpy()

    acc = accuracy_score(y_true, predicted)
    f1 = f1_score(y_true, predicted, average='weighted')

    # AUC (One-vs-Rest)
    try:
        auc = roc_auc_score(y_true, outputs.cpu().numpy(), multi_class='ovr', average='weighted')
    except:
        auc = None  # If not possible

    cm = confusion_matrix(y_true, predicted)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    if auc:
        print(f"AUC (OvR, weighted): {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    unique_labels = np.unique(np.concatenate([y_true, predicted]))
    target_names = [le.classes_[i] for i in unique_labels]
    print(classification_report(y_true, predicted, labels=unique_labels, target_names=target_names))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'data/processed/confusion_matrix_{model_name.lower().replace(" ", "_")}_{dataset}_{imbalance_mode}.png')
    plt.close()

    return {
        'accuracy': acc,
        'f1_score': f1,
        'auc': auc if auc else None
    }

# Evaluate DL models
mlp_metrics = evaluate_model(mlp, X_val_tensor, y_val_tensor, f"Advanced MLP ({imbalance_mode})")
cnn_metrics = evaluate_model(cnn, X_val_tensor, y_val_tensor, f"Advanced 1D-CNN ({imbalance_mode})")

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
    'f1_score': f1_rf,
    'auc': None  # RF doesn't have AUC in same way
}

# Save metrics to JSON
results = {
    'dataset': dataset,
    'imbalance_mode': imbalance_mode,
    'models': {
        'MLP': mlp_metrics,
        'CNN': cnn_metrics,
        'RandomForest': rf_metrics
    }
}

with open(f'data/processed/evaluation_results_{dataset}_{imbalance_mode}.json', 'w') as f:
    json.dump(results, f, indent=4)

# Feature importance plot (already done in feature_selection, but reload and show)
print(f"\nFeature importance plot already generated in feature_selection_{dataset}.py")
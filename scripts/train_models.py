import pandas as pd
import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import sys

# Make CUDA behavior deterministic and more stable on Windows; disable cuDNN if needed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# CLI
parser = argparse.ArgumentParser(description='Train models for a dataset and imbalance mode')
parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6')
parser.add_argument('imbalance_mode', type=str, choices=['zero','weighted','smote'], help='imbalance handling mode')
parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (default from TRAIN_EPOCHS env or 1)')
parser.add_argument('--train-device', type=str, choices=['cpu','cuda'], default=None, help='Training device (default cuda; required to be available)')
args = parser.parse_args()

dataset = args.dataset
imbalance_mode = args.imbalance_mode
epochs = args.epochs if args.epochs is not None else int(os.environ.get('TRAIN_EPOCHS', '1'))

# Device selection: enforce CUDA by default; error if not available when requested
train_device_env = (args.train_device or os.environ.get('TRAIN_DEVICE', 'cuda')).lower()
if train_device_env == 'cuda':
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available. Please ensure a CUDA-enabled PyTorch is installed and a GPU is available.")
    device = torch.device('cuda')
elif train_device_env == 'cpu':
    raise SystemExit("CPU training is disabled per project policy. Use --train-device cuda and ensure CUDA is available.")
else:
    # Default to strict CUDA requirement
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by default but not available.")
    device = torch.device('cuda')
print(f"Using device: {device}, Dataset: {dataset}, Imbalance: {imbalance_mode}")

# Load data
df = pd.read_csv(f'data/processed/merged_{dataset}.csv')

# Load top features
with open(f'data/processed/top_features_{dataset}.pkl', 'rb') as f:
    top_features = pickle.load(f)

# Load scaler
with open(f'data/processed/scaler_{dataset}.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Select features
X = df[top_features]
y = df['kinship']

# Scale X
X_scaled = scaler.transform(X)

# Encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Apply SMOTE if mode is smote
if imbalance_mode == 'smote':
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Original training samples: {len(y_train)}")
    print(f"SMOTE training samples: {len(y_train_smote)}")
else:
    X_train_smote, y_train_smote = X_train, y_train
    print(f"Training samples: {len(y_train)}")

# Convert to tensors (keep on CPU; move per-batch during training for stability)
X_train = torch.tensor(X_train_smote, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train_smote, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True)

# Advanced MLP Model
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

# Advanced 1D-CNN Model
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

# Train function
def train_model(model, train_loader, val_loader, y_train, imbalance_mode, epochs=1):
    # Use weighted loss only in 'weighted' mode; avoid double-compensation in 'smote'
    if imbalance_mode == 'weighted':
        # Safe weights: include all classes and avoid division by zero
        cc = torch.bincount(y_train, minlength=num_classes).to(torch.float32)
        cc = torch.clamp(cc, min=1.0)
        total = cc.sum()
        class_weights = (total / (len(cc) * cc)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model.to(device)

    def _run_epoch():
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss / max(1, len(train_loader))

    def _eval_epoch():
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        return val_loss / max(1, len(val_loader)), (100.0 * correct / max(1, total))

    for epoch in range(epochs):
        try:
            tr_loss = _run_epoch()
        except RuntimeError as e:
            if 'CUDA' in str(e).upper() and torch.backends.cudnn.enabled:
                # Retry once with cuDNN disabled to avoid Windows/cuDNN kernel issues
                print("Warning: CUDA runtime error encountered; disabling cuDNN and retrying this epoch once.")
                torch.backends.cudnn.enabled = False
                tr_loss = _run_epoch()
            else:
                raise

        scheduler.step()

        try:
            v_loss, v_acc = _eval_epoch()
        except RuntimeError as e:
            if 'CUDA' in str(e).upper() and torch.backends.cudnn.enabled:
                print("Warning: CUDA runtime error during eval; disabling cuDNN and retrying.")
                torch.backends.cudnn.enabled = False
                v_loss, v_acc = _eval_epoch()
            else:
                raise

        torch.cuda.synchronize()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {tr_loss:.4f}, Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}%")

# Train Advanced MLP
print("Training Advanced MLP...")
mlp = AdvancedMLP(len(top_features), num_classes)
train_model(mlp, train_loader, val_loader, y_train, imbalance_mode, epochs=epochs)

# Save MLP under models/<dataset>/<mode>/
models_dir = os.path.join('models', dataset, imbalance_mode)
os.makedirs(models_dir, exist_ok=True)
torch.save(mlp.state_dict(), os.path.join(models_dir, 'mlp.pth'))

# Train Advanced CNN
print("Training Advanced 1D-CNN...")
cnn = AdvancedCNN1D(len(top_features), num_classes)
train_model(cnn, train_loader, val_loader, y_train, imbalance_mode, epochs=epochs)

# Save CNN under models/<dataset>/<mode>/
torch.save(cnn.state_dict(), os.path.join(models_dir, 'cnn.pth'))

print("Models saved.")
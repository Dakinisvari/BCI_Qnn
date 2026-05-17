# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("EEG_Eye_State_Classification.csv")

X = df.drop("eyeDetection", axis=1).values
y = df["eyeDetection"].values

# =========================
# PREPROCESSING
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# pad 14 → 16 features
X_train = np.pad(X_train, ((0,0),(0,2)))
X_val   = np.pad(X_val, ((0,0),(0,2)))

# reshape → CNN input (N, C, H, W)
X_train = X_train.reshape(-1, 1, 4, 4)
X_val   = X_val.reshape(-1, 1, 4, 4)

# =========================
# TENSORS
# =========================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

# =========================
# CNN MODEL (FIXED)
# =========================
class EEG_CNN(nn.Module):
    def __init__(self):
        super(EEG_CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=2, padding=1),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = EEG_CNN().to(device)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAIN FUNCTION
# =========================
def train_cnn(model, train_loader, val_loader, epochs=15):
    for epoch in range(epochs):

        # TRAIN
        model.train()
        total_loss = 0
        correct = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()

        train_acc = correct / len(train_loader.dataset)

        # VALIDATION
        model.eval()
        correct = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()

        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
def evaluate_metrics(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, precision, recall, f1
# =========================
# RUN
# =========================
train_cnn(model, train_loader, val_loader, epochs=5)
val_acc, val_prec, val_recall, val_f1 = evaluate_metrics(model, val_loader)

print("\n===== VALIDATION METRICS =====")
print(f"Accuracy  : {val_acc:.4f}")
print(f"Precision : {val_prec:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1 Score  : {val_f1:.4f}")



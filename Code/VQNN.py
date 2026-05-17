#Variational Quantum Neural Network (VQNN)

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pennylane as qml

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

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to 4x4 (pad 14 → 16)
X_train = np.pad(X_train, ((0,0),(0,2)))
X_val   = np.pad(X_val, ((0,0),(0,2)))

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
# QUANTUM CIRCUIT
# =========================
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_net(inputs, weights):
    # Encode data
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

    # Trainable layer
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# =========================
# HYBRID MODEL
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16, 4)
        self.q_weights = nn.Parameter(0.01 * torch.randn(4))
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
      x = self.flatten(x)
      x = torch.tanh(self.fc1(x))

      q_out = []

      for i in range(x.shape[0]):
        # FIX: keep dtype consistent + preserve gradients
        q_result = quantum_net(x[i], self.q_weights)
        q_result = torch.stack(q_result).float()   # 🔥 IMPORTANT FIX

        q_out.append(q_result)

      q_out = torch.stack(q_out).to(device)

      x = self.fc2(q_out)
      return x

# =========================
# MODEL SETUP
# =========================
model = HybridModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =========================
# TRAINING
# =========================
def train_model(model, epochs=15):
    for epoch in range(epochs):
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

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")

# =========================
# TRAIN
# =========================
train_model(model, epochs=20)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# EVALUATION
# =========================
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# =========================
# METRICS
# =========================
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print("\n===== VQNN PERFORMANCE =====")
print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)

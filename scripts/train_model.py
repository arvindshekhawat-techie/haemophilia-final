import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.elbow_lstm import ElbowLSTM

FEATURE_DIR = Path("features")

X, y = [], []

print("📂 Loading features...")

for label, val in [("correct", 1), ("incorrect", 0)]:
    files = list((FEATURE_DIR / label).glob("*.npy"))
    print(f"{label}: {len(files)} files")

    for f in files:
        X.append(np.load(f))
        y.append(val)

# =========================
# CHECK DATA
# =========================
if len(X) == 0:
    raise ValueError("❌ No training data found. Check features folder.")

X = np.array(X)
y = np.array(y)

print(f"\nDataset shape: {X.shape}")

# =========================
# NORMALIZATION (VERY IMPORTANT)
# =========================
mean = X.mean(axis=(0, 1), keepdims=True)
std = X.std(axis=(0, 1), keepdims=True) + 1e-6

X = (X - mean) / std

# 🔥 Save normalization (IMPORTANT for realtime)
np.save("models/mean.npy", mean)
np.save("models/std.npy", std)

# =========================
# TORCH TENSORS
# =========================
X = torch.tensor(X).float()
y = torch.tensor(y).float().unsqueeze(1)

# =========================
# SPLIT
# =========================
dataset = TensorDataset(X, y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# =========================
# MODEL
# =========================
model = ElbowLSTM(input_size=X.shape[2])

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 🔥 CLASS IMBALANCE FIX
pos_weight = torch.tensor([len(y[y == 0]) / (len(y[y == 1]) + 1e-6)])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# =========================
# TRAIN
# =========================
EPOCHS = 30
best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        out = model(xb)
        loss = criterion(out, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()

            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total

    print(f"Epoch {epoch+1}: Train {train_loss:.3f} | Val {val_loss:.3f} | Acc {acc:.2f}")

    # 🔥 SAVE BEST MODEL ONLY
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/elbow_lstm_v4.pth")
        print("💾 Saved best model")

print("\n🔥 Training complete")
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from src.training.dataset import ElbowDataset
from src.models.elbow_lstm import ElbowLSTM
from src.features.elbow_features import INPUT_SIZE


def find_dataset(root):
    """Auto find valid processed dataset"""
    candidates = [
        root / "processed_clean_v3",
        root / "processed_clean",
        root / "processed_debug",
        root / "processed_rebuilt_v2",
    ]

    for path in candidates:
        if path.exists() and any(path.rglob("*.npy")):
            print(f"✅ Using dataset: {path}")
            return path

    raise ValueError("❌ No processed dataset found. Run preprocessing first.")


def train():
    print("🔥 TRAINING STARTED")

    root = Path(__file__).resolve().parents[2]

    # =========================
    # FIND DATASET
    # =========================
    processed_dir = find_dataset(root)

    dataset = ElbowDataset(processed_dir)

    print(f"📊 Dataset size: {len(dataset)}")

    if len(dataset) < 20:
        raise ValueError("❌ Too few samples. Preprocessing failed.")

    # =========================
    # DEVICE
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⚡ Device:", device)

    # =========================
    # CLASS BALANCE (🔥 FIX)
    # =========================
    labels = []

    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y.item()))

    labels = np.array(labels)

    num_pos = np.sum(labels)
    num_neg = len(labels) - num_pos

    print(f"✅ Positives: {num_pos}, Negatives: {num_neg}")

    pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)]).to(device)

    # =========================
    # SPLIT
    # =========================
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    # =========================
    # MODEL
    # =========================
    model = ElbowLSTM(input_size=INPUT_SIZE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 🔥 FIXED LOSS
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(30):
        print(f"\n🚀 Epoch {epoch+1}")

        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # =====================
        # VALIDATION
        # =====================
        model.eval()
        preds, actual = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)

                out = model(x)
                probs = torch.sigmoid(out)

                p = (probs > 0.5).int().cpu().numpy()

                preds.extend(p.flatten())
                actual.extend(y.numpy())

        acc = accuracy_score(actual, preds)
        f1 = f1_score(actual, preds, zero_division=0)

        print(f"📉 Train Loss: {train_loss:.4f}")
        print(f"🎯 Accuracy: {acc:.4f}")
        print(f"🔥 F1 Score: {f1:.4f}")

        # =====================
        # SAVE BEST
        # =====================
        if train_loss < best_loss:
            best_loss = train_loss

            model_path = root / "models" / "elbow_lstm_v3.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), model_path)

            print("✅ Model Saved")

    print("\n🎯 TRAINING COMPLETE")


if __name__ == "__main__":
    train()
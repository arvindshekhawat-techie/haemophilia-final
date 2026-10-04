from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

from src.features.elbow_features import INPUT_SIZE, SEQUENCE_LENGTH


class ElbowDataset(Dataset):
    def __init__(self, processed_dir):
        self.files = sorted(Path(processed_dir).rglob("*.npy"))

        if not self.files:
            raise ValueError("❌ No data found")

    @property
    def groups(self):
        return [str(f.parent) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = np.load(self.files[idx], allow_pickle=True).item()

        x = np.array(item["features"], dtype=np.float32)
        y = float(item["label"])

        # =========================
        # 🔥 FIX: HANDLE VARIABLE LENGTH
        # =========================
        if x.shape[0] < SEQUENCE_LENGTH:
            # pad
            pad_len = SEQUENCE_LENGTH - x.shape[0]
            pad = np.zeros((pad_len, INPUT_SIZE), dtype=np.float32)
            x = np.vstack([x, pad])

        elif x.shape[0] > SEQUENCE_LENGTH:
            # trim (keep last part → most important)
            x = x[-SEQUENCE_LENGTH:]

        # final safety check
        if x.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(f"Still bad shape: {x.shape}")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
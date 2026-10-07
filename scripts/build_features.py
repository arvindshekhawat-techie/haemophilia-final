import numpy as np
from pathlib import Path
from src.features.elbow_features import SEQUENCE_LENGTH

INPUT = Path("processed")
OUTPUT = Path("features")


def build(seq):
    seq = np.array(seq, dtype=np.float32)

    if len(seq) < 5:
        return None

    elbow = seq[:, 0]
    shoulder = seq[:, 1]
    wrist = seq[:, 2]
    coords = seq[:, 3:]

    velocity = np.gradient(elbow) if len(elbow) > 1 else np.zeros_like(elbow)
    acceleration = np.gradient(velocity) if len(elbow) > 2 else np.zeros_like(elbow)

    velocity = velocity / (np.max(np.abs(velocity)) + 1e-6)
    acceleration = acceleration / (np.max(np.abs(acceleration)) + 1e-6)

    rom = np.ptp(elbow) + 1e-6

    smoothness = np.std(np.diff(elbow)) if len(elbow) > 1 else 0.0
    speed_std = np.std(velocity)
    peak = np.max(elbow)

    extension_deficit = (180 - peak) / 180.0
    flexion_deficit = np.min(elbow) / 180.0

    features = []

    for i in range(len(seq)):
        t = i / len(seq)

        direction = 1.0 if velocity[i] > 0 else -1.0
        pause = 1.0 if abs(velocity[i]) < 0.05 else 0.0
        completion = (elbow[i] - np.min(elbow)) / rom

        features.append([
            elbow[i] / 180.0,
            velocity[i],
            acceleration[i],
            completion,
            direction,
            pause,
            rom,
            smoothness,
            speed_std,
            peak,
            t,
            extension_deficit,
            flexion_deficit,
            shoulder[i] / 180.0,
            wrist[i] / 180.0,
            *coords[i]
        ])

    features = np.array(features, dtype=np.float32)

    if len(features) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(features), features.shape[1]), dtype=np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[-SEQUENCE_LENGTH:]

    return features


total = 0

for label in ["correct", "incorrect"]:
    input_folder = INPUT / label
    output_folder = OUTPUT / label

    output_folder.mkdir(parents=True, exist_ok=True)

    files = list(input_folder.glob("*.npy"))
    print(f"\n📂 {label}: {len(files)} files")

    for file in files:
        try:
            seq = np.load(file)

            feat = build(seq)
            if feat is None:
                continue

            np.save(output_folder / file.name, feat)
            total += 1

        except Exception as e:
            print(f"❌ Error: {file.name} -> {e}")

print(f"\n✅ Features built: {total}")
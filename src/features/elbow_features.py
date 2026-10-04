"""
Advanced biomechanical features for elbow flexion (v3)
"""

import numpy as np

SEQUENCE_LENGTH = 25
INPUT_SIZE = 14
FEATURE_SCHEMA_VERSION = 3

EPS = 1e-6

# Landmarks (MediaPipe)
SHOULDER = 12
ELBOW = 14
WRIST = 16
HIP = 24


def get_lm(row, i):
    return np.array([
        row.get(f"landmark_{i}_x", 0.0),
        row.get(f"landmark_{i}_y", 0.0),
        row.get(f"landmark_{i}_z", 0.0),
    ], dtype=np.float32)


def angle(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + EPS)
    v2 = v2 / (np.linalg.norm(v2) + EPS)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


def elbow_angle(row):
    s = get_lm(row, SHOULDER)
    e = get_lm(row, ELBOW)
    w = get_lm(row, WRIST)
    return angle(s - e, w - e)


def compute_speed(points, fps, scale):
    diff = np.diff(points, axis=0, prepend=points[:1])
    return np.linalg.norm(diff, axis=1) * fps / (scale + EPS)


def compute_jerk(signal, fps):
    vel = np.gradient(signal) * fps
    acc = np.gradient(vel) * fps
    jerk = np.gradient(acc) * fps
    return jerk


def resample(seq, n=SEQUENCE_LENGTH):
    if len(seq) < 2:
        return np.repeat(seq, n, axis=0)

    x = np.linspace(0, len(seq) - 1, n)
    xp = np.arange(len(seq))

    return np.column_stack([
        np.interp(x, xp, seq[:, i]) for i in range(seq.shape[1])
    ])


def build_features(rows, fps, angle_series=None):
    
    if angle_series is None:
        raise ValueError("angle_series required")

    # ✅ convert to numpy
    angles = np.array(angle_series, dtype=np.float32)

    if len(angles) < 5:
        raise ValueError("Too few frames")

    # =========================
    # CORE SIGNALS
    # =========================
    velocity = np.gradient(angles)
    acceleration = np.gradient(velocity)

    # ✅ NORMALIZATION
    velocity = velocity / (np.max(np.abs(velocity)) + 1e-6)
    acceleration = acceleration / (np.max(np.abs(acceleration)) + 1e-6)

    # =========================
    # BIOMECHANICS
    # =========================
    rom = np.max(angles) - np.min(angles) + 1e-6
    angle_norm = angles / 180.0
    completion = (angles - np.min(angles)) / rom

    smoothness = np.std(np.diff(angles)) if len(angles) > 1 else 0.0
    speed_std = np.std(velocity)
    peak = np.max(angles)

    pause = (np.abs(velocity) < 0.05).astype(float)

    # =========================
    # BUILD FEATURES
    # =========================
    features = []

    for i in range(len(angles)):

        t = i / len(angles)

        direction = 1.0 if velocity[i] > 0 else -1.0

        features.append([
            angle_norm[i],        # 1
            velocity[i],          # 2
            acceleration[i],      # 3
            completion[i],        # 4
            direction,            # 5
            pause[i],             # 6
            rom,                  # 7
            smoothness,           # 8
            speed_std,            # 9
            peak,                 # 10
            t,                    # 11
            0.0, 0.0, 0.0         # padding → total = 14
        ])

    return np.array(features, dtype=np.float32), None

def smooth_angles(a):
    if len(a) < 3:
        return a
    return np.convolve(a, np.ones(5)/5, mode="same")


def filter_angle_outliers(a):
    m = np.median(a)
    return np.clip(a, m - 60, m + 60)
import numpy as np

SEQUENCE_LENGTH = 30
INPUT_SIZE = 22   # 🔥 UPDATED


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def extract_angles(row):
    shoulder = row["shoulder"]
    elbow = row["elbow"]
    wrist = row["wrist"]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # shoulder angle (approx)
    shoulder_angle = calculate_angle(elbow, shoulder, shoulder + np.array([0, -0.1]))

    # wrist pseudo angle
    wrist_angle = calculate_angle(elbow, wrist, wrist + np.array([0.1, 0]))

    # arm length (scale normalization)
    arm_length = np.linalg.norm(shoulder - wrist)

    return {
        "elbow_angle": elbow_angle,
        "shoulder_angle": shoulder_angle,
        "wrist_angle": wrist_angle,
        "elbow": elbow,
        "wrist": wrist,
        "shoulder": shoulder,
        "arm_length": arm_length
    }
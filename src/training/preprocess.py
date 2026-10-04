"""
Robust preprocessing (FINAL FIXED)

✔ Detects reps reliably
✔ Does NOT over-filter
✔ Gives debug output
✔ Saves usable data
"""

from math import ceil
from pathlib import Path
import numpy as np

from src.pose.pose_extraction import extract_video_landmarks, video_fps
from src.features.elbow_features import (
    build_features,
    elbow_angle,
    filter_angle_outliers,
    smooth_angles,
    FEATURE_SCHEMA_VERSION,
)
from src.exercises.elbow_flexion import detect_rep_boundaries


# ============================
# CONFIG (VERY IMPORTANT)
# ============================
TARGET_FPS = 15.0

MIN_ROM = 8              # 🔥 VERY RELAXED
MIN_FRAMES = 6           # 🔥 VERY RELAXED
MAX_VISIBILITY_DROP = 0.7  # 🔥 VERY RELAXED


# ============================
def preprocess_dataset(dataset_dir, processed_dir):
    dataset_dir = Path(dataset_dir)
    processed_dir = Path(processed_dir)

    if processed_dir.exists():
        print("❌ Delete processed folder first")
        return

    for class_name, label in [("correct", 1), ("incorrect", 0)]:

        class_path = dataset_dir / class_name
        save_dir = processed_dir / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(class_path.glob("*.mp4"))

        for video_path in videos:
            print(f"\n🎥 Processing: {video_path.name}")

            fps_original = video_fps(video_path)
            stride = max(1, ceil(fps_original / TARGET_FPS))

            rows = extract_video_landmarks(video_path, frame_stride=stride)

            if not rows:
                print("❌ No landmarks")
                continue

            fps = fps_original / stride

            # -------------------------
            # ANGLES
            # -------------------------
            raw_angles = np.array([elbow_angle(r) for r in rows], dtype=np.float32)
            angles = smooth_angles(filter_angle_outliers(raw_angles))

            print(f"Angle range: {angles.min():.2f} → {angles.max():.2f}")

            # -------------------------
            # DETECT REPS
            # -------------------------
            boundaries = detect_rep_boundaries(angles, fps=fps)

            print(f"Detected reps: {len(boundaries)}")

            rep_count = 0

            for i, (start, end) in enumerate(boundaries):

                rep_rows = rows[start:end + 1]
                rep_angles = angles[start:end + 1]

                print(f"\n--- REP {i+1} ---")

                # -------------------------
                # VISIBILITY
                # -------------------------
                bad_frames = sum(
                    any(
                        row.get(f"landmark_{j}_visibility", 0) < 0.5
                        for j in [12, 14, 16]
                    )
                    for row in rep_rows
                )

                visibility_ratio = bad_frames / max(len(rep_rows), 1)
                print(f"Visibility: {visibility_ratio:.2f}")

                if visibility_ratio > MAX_VISIBILITY_DROP:
                    print("❌ Skipped: visibility")
                    continue

                # -------------------------
                # RANGE OF MOTION
                # -------------------------
                rom = float(np.ptp(rep_angles))
                print(f"ROM: {rom:.2f}")

                if rom < MIN_ROM:
                    print("❌ Skipped: low ROM")
                    continue

                # -------------------------
                # LENGTH
                # -------------------------
                print(f"Frames: {len(rep_angles)}")

                if len(rep_angles) < MIN_FRAMES:
                    print("❌ Skipped: too short")
                    continue

                # -------------------------
                # BUILD FEATURES
                # -------------------------
                features, _ = build_features(
                    rep_rows,
                    fps,
                    angle_series=rep_angles,
                )

                # -------------------------
                # SAVE
                # -------------------------
                save_path = save_dir / f"{video_path.stem}_rep{rep_count}.npy"

                np.save(
                    save_path,
                    {
                        "features": features,
                        "label": label,
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "range": rom,
                        "duration": float(len(rep_angles) / fps),
                        "peak": float(np.max(rep_angles)),
                    }
                )

                print("✅ Saved")
                rep_count += 1

            print(f"🎯 Final saved reps: {rep_count}")


# ============================
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    dataset_dir = root / "dataset" / "elbow_flexion"
    processed_dir = root / "processed_debug"

    preprocess_dataset(dataset_dir, processed_dir)
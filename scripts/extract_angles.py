import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

from src.pose.pose_extraction import extract_frame_landmarks
from src.features.elbow_features import extract_angles

mp_pose = mp.solutions.pose


# =========================
# PATH FIX (VERY IMPORTANT)
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

DATASET = BASE_DIR / "dataset"
OUTPUT = BASE_DIR / "processed"

print(f"📂 Dataset path: {DATASET}")
print(f"💾 Output path: {OUTPUT}")


# =========================
# VIDEO PROCESSOR
# =========================
def process_video(video_path, out_path):
    print(f"▶ Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    pose = mp_pose.Pose()

    angles_seq = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        row, _ = extract_frame_landmarks(frame, pose)
        if row is None:
            continue

        data = extract_angles(row)

        angles_seq.append([
            data["elbow_angle"],
            data["shoulder_angle"],
            data["wrist_angle"],
            *data["elbow"],
            *data["wrist"],
            *data["shoulder"],
            data["arm_length"]
        ])

    cap.release()
    pose.close()

    # =========================
    # SAVE CHECK
    # =========================
    if len(angles_seq) > 10:
        np.save(out_path, np.array(angles_seq))
        print(f"✅ Saved: {out_path.name} | Frames: {len(angles_seq)}")
    else:
        print(f"⚠ Skipped (too small): {video_path.name} | Frames: {len(angles_seq)}")


# =========================
# MAIN LOOP
# =========================
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

for label in ["correct", "incorrect"]:

    input_folder = DATASET / label
    output_folder = OUTPUT / label

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Checking folder: {input_folder}")

    if not input_folder.exists():
        print(f"❌ Folder not found: {input_folder}")
        continue

    videos = list(input_folder.glob("*"))

    if len(videos) == 0:
        print("⚠ No files found")
        continue

    for vid in videos:

        if not vid.suffix.lower() in VIDEO_EXTS:
            continue

        out_file = output_folder / (vid.stem + ".npy")

        process_video(vid, out_file)

print("\n✅ Angle extraction complete.")
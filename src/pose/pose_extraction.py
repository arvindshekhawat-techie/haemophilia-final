"""MediaPipe Pose extraction for offline videos and live frames."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp


# ===============================
# MEDIAPIPE INIT (ONLY ONCE)
# ===============================
mp_pose = mp.solutions.pose

# Do NOT recreate inside loop
# We will pass this from outside


# ===============================
# FRAME LANDMARK EXTRACTION
# ===============================
def extract_frame_landmarks(frame, pose) -> Tuple[Optional[Dict[str, float]], object]:
    """
    Extract 33 pose landmarks from a single frame.

    Args:
        frame: BGR frame from OpenCV
        pose: initialized MediaPipe Pose object

    Returns:
        landmark dictionary or None, plus the MediaPipe results object.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return None, results

    return _landmark_row(results.pose_landmarks), results


def extract_video_landmarks(
    video_path: str | Path,
    frame_stride: int = 1,
) -> List[Dict[str, float]]:
    """Extract pose landmarks at a fixed frame stride from a readable video."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    rows = []
    with mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_index = 0
        while True:
            success = video.grab()
            if not success:
                break
            if frame_index % frame_stride:
                frame_index += 1
                continue
            success, frame = video.retrieve()
            if not success:
                frame_index += 1
                continue
            row, _ = extract_frame_landmarks(frame, pose)
            rows.append(row or _empty_landmark_row())
            frame_index += 1
    video.release()
    if not rows:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    return rows


def video_fps(video_path: str | Path, default: float = 30.0) -> float:
    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return float(fps) if fps > 0 else default


# ===============================
# INTERNAL: FORMAT LANDMARKS
# ===============================
def _landmark_row(landmarks) -> Dict[str, float]:
    row: Dict[str, float] = {}

    for i in range(33):
        lm = landmarks.landmark[i]

        row[f"landmark_{i}_x"] = float(lm.x)
        row[f"landmark_{i}_y"] = float(lm.y)
        row[f"landmark_{i}_z"] = float(lm.z)
        row[f"landmark_{i}_visibility"] = float(lm.visibility)

    return row


def _empty_landmark_row() -> Dict[str, float]:
    return {
        f"landmark_{index}_{axis}": 0.0
        for index in range(33)
        for axis in ("x", "y", "z", "visibility")
    }
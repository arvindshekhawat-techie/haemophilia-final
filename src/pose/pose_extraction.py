import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose


# ============================================================
# EXTRACT LANDMARKS
# ============================================================

def extract_frame_landmarks(frame, pose):
    """
    Process one video frame using MediaPipe Pose.

    Returns:
        row:
            Dictionary containing right-side shoulder, elbow,
            and wrist coordinates.

        results:
            Complete MediaPipe Pose result. This is used by the
            realtime system for drawing the body wireframe.
    """

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark

    def get_point(idx):
        return np.array(
            [landmarks[idx].x, landmarks[idx].y],
            dtype=np.float32
        )

    # ========================================================
    # RIGHT SIDE
    #
    # MediaPipe Pose:
    # 12 -> Right Shoulder
    # 14 -> Right Elbow
    # 16 -> Right Wrist
    # ========================================================

    shoulder = get_point(12)
    elbow = get_point(14)
    wrist = get_point(16)

    # ========================================================
    # VISIBILITY
    # ========================================================

    shoulder_visibility = landmarks[12].visibility
    elbow_visibility = landmarks[14].visibility
    wrist_visibility = landmarks[16].visibility

    row = {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,

        "shoulder_visibility": shoulder_visibility,
        "elbow_visibility": elbow_visibility,
        "wrist_visibility": wrist_visibility
    }

    return row, results
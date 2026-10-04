"""Real-time elbow inference — FINAL (ANGLE RANGE BASED FORM FIX)"""

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.features.elbow_features import build_features
from src.inference.inference import ElbowInference
from src.pose.pose_extraction import extract_frame_landmarks


# ================== TUNING ==================
ANGLE_EMA_ALPHA = 0.25
ANGLE_DEADZONE = 1.0

CALIBRATION_FRAMES = 40

MIN_REP_FRAMES = 10
REP_COOLDOWN = 6

DIRECTION_SMOOTH = 4

UP_TRIGGER_OFFSET = 20
DOWN_TRIGGER_OFFSET = 12
# ===========================================


class LiveRepTracker:
    def __init__(self):
        self.ema_angle = None
        self.last_angle = None
        self.prev_angle = None

        self.direction_buffer = deque(maxlen=DIRECTION_SMOOTH)
        self.direction = None

        self.state = "DOWN"

        self.calibration = []
        self.calibrated = False
        self.min_angle = None
        self.max_angle = None

        self.up_trigger = None
        self.down_trigger = None

        self.rep_count = 0
        self.rep_frames = 0
        self.cooldown = 0

        self.rep_rows = []
        self.rep_angles = []

        self.form = "WAITING"

    def smooth_angle(self, shoulder, elbow, wrist):
        a, b, c = np.array(shoulder), np.array(elbow), np.array(wrist)

        ba = a - b
        bc = c - b

        angle = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
                    -1, 1
                )
            )
        )

        angle = float(np.clip(angle, 40, 170))

        if self.ema_angle is None:
            self.ema_angle = angle
        else:
            self.ema_angle = ANGLE_EMA_ALPHA * angle + (1 - ANGLE_EMA_ALPHA) * self.ema_angle

        if self.last_angle is not None and abs(self.ema_angle - self.last_angle) < ANGLE_DEADZONE:
            return self.last_angle

        self.last_angle = self.ema_angle
        return self.last_angle

    def update(self, row, predictor=None):
        if row is None:
            return self.result()

        shoulder = _p(row, 12)
        elbow = _p(row, 14)
        wrist = _p(row, 16)

        angle = self.smooth_angle(shoulder, elbow, wrist)

        if self.prev_angle is not None:
            if angle < self.prev_angle:
                self.direction_buffer.append("UP")
            elif angle > self.prev_angle:
                self.direction_buffer.append("DOWN")

        self.prev_angle = angle

        if self.direction_buffer:
            self.direction = max(set(self.direction_buffer), key=self.direction_buffer.count)

        # ===== CALIBRATION =====
        if not self.calibrated:
            self.calibration.append(angle)

            if len(self.calibration) >= CALIBRATION_FRAMES:
                mn, mx = min(self.calibration), max(self.calibration)

                if mx - mn > 15:
                    self.min_angle = mn
                    self.max_angle = mx

                    self.up_trigger = mx - UP_TRIGGER_OFFSET
                    self.down_trigger = mx - DOWN_TRIGGER_OFFSET

                    print("CALIBRATED:", mn, mx)

                    self.calibrated = True

            return self.result()

        # ===== STATE MACHINE (UNCHANGED) =====
        if self.direction == "UP" and angle < self.up_trigger:
            if self.state != "UP":
                self.state = "UP"
                self.rep_frames = 0
                self.rep_rows = []
                self.rep_angles = []

        elif self.direction == "DOWN" and angle > self.down_trigger:

            if self.state == "UP" and self.rep_frames > MIN_REP_FRAMES and self.cooldown == 0:

                self.rep_count += 1

                self.rep_rows.append(row)
                self.rep_angles.append(angle)

                if predictor is None:
                    self.form = "Uncertain"
                else:
                    features, _ = build_features(
                        self.rep_rows,
                        30,
                        angle_series=np.asarray(self.rep_angles, dtype=np.float32),
                        feature_schema_version=(
                            predictor.feature_schema_version
                            if predictor
                            else 2
                        ),
                    )
                    prediction, _ = predictor.predict(features)
                    self.form = {
                        "CORRECT": "Correct",
                        "INCORRECT": "Incorrect",
                        "UNCERTAIN": "Uncertain",
                    }[prediction]

                self.cooldown = REP_COOLDOWN

            self.state = "DOWN"

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.state == "UP":
            self.rep_frames += 1
            self.rep_rows.append(row)
            self.rep_angles.append(angle)

        return self.result()

    def result(self):
        return {
            "state": self.state,
            "reps": self.rep_count,
            "form": self.form,
            "angle": self.last_angle or 0,
        }


def run_live_camera(model_path):
    predictor = ElbowInference(model_path)
    tracker = LiveRepTracker()

    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            row, res = extract_frame_landmarks(frame, pose)

            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            r = tracker.update(row, predictor)
            _draw(frame, r)

            cv2.imshow("Elbow AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def _draw(frame, r):
    if r["form"] in ("Correct", "CORRECT"):
        color = (0, 255, 0)
    elif r["form"] in ("Incorrect", "INCORRECT"):
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)

    cv2.putText(frame, f"Reps: {r['reps']}", (20, 40), 0, 0.8, color, 2)
    cv2.putText(frame, f"Form: {r['form']}", (20, 70), 0, 0.7, color, 2)
    cv2.putText(frame, f"Angle: {r['angle']:.1f}", (20, 100), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"State: {r['state']}", (20, 130), 0, 0.7, (255,255,255), 2)
    cv2.putText(
        frame,
        "EXPERIMENTAL - NOT FOR MEDICAL USE",
        (20, 180),
        0,
        0.6,
        (0, 165, 255),
        2,
    )


def _p(row, i):
    return np.array([
        row.get(f"landmark_{i}_x",0),
        row.get(f"landmark_{i}_y",0),
        row.get(f"landmark_{i}_z",0),
    ])


if __name__ == "__main__":
    run_live_camera(Path(__file__).resolve().parents[2] / "models" / "elbow_lstm.pth")
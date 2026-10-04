from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
import mediapipe as mp

from src.pose.pose_extraction import extract_frame_landmarks, mp_pose
from src.features.elbow_features import elbow_angle, SEQUENCE_LENGTH, INPUT_SIZE
from src.models.elbow_lstm import ElbowLSTM


MODEL_PATH = Path("models/elbow_lstm_v3.pth")

CALIBRATION_SECONDS = 2
EMA_ALPHA = 0.4

DISPLAY_HOLD_FRAMES = 20


def smooth_angle(angle, prev):
    if prev is None:
        return angle, angle
    ema = EMA_ALPHA * angle + (1 - EMA_ALPHA) * prev
    return ema, ema


def build_live_features(angle_series):

    angles = np.array(angle_series, dtype=np.float32)

    velocity = np.gradient(angles)
    acceleration = np.gradient(velocity)

    velocity = velocity / (np.max(np.abs(velocity)) + 1e-6)
    acceleration = acceleration / (np.max(np.abs(acceleration)) + 1e-6)

    rom = np.max(angles) - np.min(angles) + 1e-6

    angle_norm = angles / 180.0
    completion = (angles - np.min(angles)) / rom

    smoothness = np.std(np.diff(angles)) if len(angles) > 1 else 0.0
    speed_std = np.std(velocity)
    peak = np.max(angles)

    pause = (np.abs(velocity) < 0.05).astype(float)

    extension_deficit = (180 - np.max(angles)) / 180.0
    flexion_deficit = (np.min(angles)) / 180.0

    features = []

    for i in range(len(angles)):
        t = i / len(angles)
        direction = 1.0 if velocity[i] > 0 else -1.0

        features.append([
            angle_norm[i],
            velocity[i],
            acceleration[i],
            completion[i],
            direction,
            pause[i],
            rom,
            smoothness,
            speed_std,
            peak,
            t,
            extension_deficit,
            flexion_deficit,
            0.0
        ])

    features = np.array(features, dtype=np.float32)

    if len(features) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(features), INPUT_SIZE), dtype=np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[-SEQUENCE_LENGTH:]

    return features


class RepDetector:
    def __init__(self, fps):
        self.state = "IDLE"
        self.start = None
        self.last_rep_frame = -1000
        self.cooldown = int(0.4 * fps)
        self.fps = fps

        self.vel_buffer = deque(maxlen=3)

        self.peak_angle = None
        self.reached_top = False

    def update(self, angle, velocity, idx):

        self.vel_buffer.append(velocity)
        vel = np.mean(self.vel_buffer)

        if vel > 0.4:
            movement = "UP"
        elif vel < -0.4:
            movement = "DOWN"
        else:
            movement = "HOLD"

        if idx - self.last_rep_frame < self.cooldown:
            return None, movement

        if self.state == "IDLE":
            if movement == "DOWN":
                self.state = "DOWN"
                self.start = idx
                self.peak_angle = angle
                self.reached_top = False

        elif self.state == "DOWN":
            if movement == "UP":
                self.state = "UP"
                self.peak_angle = angle

        elif self.state == "UP":

            if angle > self.peak_angle:
                self.peak_angle = angle

            if angle < self.peak_angle - 10:
                self.reached_top = True

            if movement == "DOWN" and self.reached_top:
                duration = idx - self.start

                if duration > int(0.4 * self.fps):
                    self.last_rep_frame = idx
                    rep = (self.start, idx)

                    self.state = "DOWN"
                    self.start = idx
                    self.peak_angle = angle
                    self.reached_top = False

                    return rep, movement

        return None, movement


def load_model():
    model = ElbowLSTM(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def run():

    model = load_model()

    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30

    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    prev_angle = None
    prev_ema = None

    all_angles = []
    calibration = []
    calibrated = False

    detector = None

    rep_count = 0
    label = "CALIBRATING"
    movement = "IDLE"

    pred_buffer = deque(maxlen=4)

    display_label = "CALIBRATING"
    display_timer = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        row, _ = extract_frame_landmarks(frame, pose)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if row is not None:

            raw = elbow_angle(row)
            angle, prev_ema = smooth_angle(raw, prev_ema)

            all_angles.append(angle)

            velocity = angle - prev_angle if prev_angle is not None else 0
            prev_angle = angle

            if not calibrated:
                calibration.append(angle)

                if len(calibration) > fps * CALIBRATION_SECONDS:
                    calibrated = True
                    detector = RepDetector(fps)
                    label = "READY"
                    display_label = "READY"

            else:
                rep, movement = detector.update(angle, velocity, len(all_angles) - 1)

                if display_timer == 0 and movement == "UP":
                    display_label = "MOVING"

                if rep is not None:
                    s, e = rep

                    rep_angles = all_angles[s:e+1]
                    duration = len(rep_angles) / fps
                    rom = np.ptp(rep_angles)

                    if duration < 0.35 or rom < 20:
                        continue

                    features = build_live_features(rep_angles)

                    with torch.no_grad():
                        prob = torch.sigmoid(
                            model(torch.from_numpy(features[None]).float())
                        ).item()

                    pred_buffer.append(prob)
                    smooth_prob = np.mean(pred_buffer)

                    print(f"Prob: {smooth_prob:.3f}")

                    # 🔥 IMPROVED DECISION LOGIC
                    if smooth_prob > 0.75:
                        label = "CORRECT"

                    elif smooth_prob < 0.40:
                        label = "INCORRECT"

                    else:
                        # 🔥 Strong fallback logic
                        if rom > 40 and duration > 0.6:
                            label = "CORRECT"
                        else:
                            label = "INCORRECT"

                    rep_count += 1

                    display_label = label
                    display_timer = DISPLAY_HOLD_FRAMES

        if display_timer > 0:
            display_timer -= 1

        if display_label == "CORRECT":
            color = (0, 255, 0)
        elif display_label == "INCORRECT":
            color = (0, 0, 255)
        elif display_label == "READY":
            color = (0, 255, 255)
        else:
            color = (255, 255, 255)

        cv2.putText(frame, f"Reps: {rep_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Status: {display_label}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"State: {movement}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Elbow AI", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
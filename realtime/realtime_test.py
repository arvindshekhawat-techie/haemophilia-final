from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pose.pose_extraction import (
    extract_frame_landmarks,
    mp_pose
)

from src.features.elbow_features import (
    extract_angles,
    SEQUENCE_LENGTH
)

from src.models.elbow_lstm import ElbowLSTM


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("models/elbow_lstm_v4.pth")
MEAN_PATH = Path("models/mean.npy")
STD_PATH = Path("models/std.npy")


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

CALIBRATION_SECONDS = 2

EMA_ALPHA = 0.35

MIN_VISIBILITY = 0.55


# ============================================================
# REP DETECTION
# ============================================================

# Minimum movement required before a repetition can begin.
MIN_REP_ROM = 30.0

# How close the arm must return to the starting DOWN position.
BASELINE_TOLERANCE = 18.0

# Minimum duration of one complete repetition.
MIN_REP_SECONDS = 0.35

# Safety timeout.
MAX_REP_SECONDS = 15.0

# Minimum frame-to-frame movement considered meaningful.
MOVEMENT_THRESHOLD = 0.18

# Velocity smoothing.
VELOCITY_BUFFER_SIZE = 5

# Small cooldown after a completed repetition.
REP_COOLDOWN_SECONDS = 0.10


# ============================================================
# MODEL SETTINGS
# ============================================================

AI_CORRECT_THRESHOLD = 0.60

AI_REJECT_THRESHOLD = 0.25


# ============================================================
# MOVEMENT QUALITY SETTINGS
# ============================================================

# ROM
ROM_EXCELLENT = 120.0
ROM_GOOD = 90.0
ROM_ACCEPTABLE = 60.0
ROM_MINIMUM = 30.0


# Shoulder movement
SHOULDER_GOOD = 0.030
SHOULDER_ACCEPTABLE = 0.060
SHOULDER_BAD = 0.090


# Smoothness
SMOOTHNESS_GOOD = 12.0
SMOOTHNESS_ACCEPTABLE = 20.0
SMOOTHNESS_BAD = 30.0


# Duration
REP_DURATION_FAST = 0.45
REP_DURATION_SLOW = 7.0


# ============================================================
# SCORE WEIGHTS
# ============================================================

WEIGHT_ROM = 0.35
WEIGHT_SHOULDER = 0.20
WEIGHT_SMOOTHNESS = 0.15
WEIGHT_AI = 0.30


# ============================================================
# COLORS
# ============================================================

GREEN = (0, 220, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
BLUE = (255, 150, 0)
GRAY = (170, 170, 170)
BLACK = (0, 0, 0)


# ============================================================
# ANGLE SMOOTHING
# ============================================================

def smooth_angle(angle, previous):

    if previous is None:
        return float(angle), float(angle)

    smoothed = (
        EMA_ALPHA * angle
        + (1.0 - EMA_ALPHA) * previous
    )

    return float(smoothed), float(smoothed)


# ============================================================
# MODEL FEATURE BUILDER
# MUST MATCH TRAINING
# ============================================================

def build_features(seq):

    seq = np.array(
        seq,
        dtype=np.float32
    )

    if len(seq) < 5:
        return None

    elbow = seq[:, 0]
    shoulder = seq[:, 1]
    wrist = seq[:, 2]

    coords = seq[:, 3:]

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    velocity = (
        np.gradient(elbow)
        if len(elbow) > 1
        else np.zeros_like(elbow)
    )

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    acceleration = (
        np.gradient(velocity)
        if len(elbow) > 2
        else np.zeros_like(elbow)
    )

    # --------------------------------------------------------
    # Normalize velocity
    # --------------------------------------------------------

    velocity = velocity / (
        np.max(np.abs(velocity)) + 1e-6
    )

    # --------------------------------------------------------
    # Normalize acceleration
    # --------------------------------------------------------

    acceleration = acceleration / (
        np.max(np.abs(acceleration)) + 1e-6
    )

    # --------------------------------------------------------
    # ROM
    # --------------------------------------------------------

    rom = np.ptp(elbow) + 1e-6

    # --------------------------------------------------------
    # Smoothness
    # --------------------------------------------------------

    smoothness = (
        np.std(np.diff(elbow))
        if len(elbow) > 1
        else 0.0
    )

    # --------------------------------------------------------
    # Speed variation
    # --------------------------------------------------------

    speed_std = np.std(velocity)

    # --------------------------------------------------------
    # Peak
    # --------------------------------------------------------

    peak = np.max(elbow)

    # --------------------------------------------------------
    # Extension deficit
    # --------------------------------------------------------

    extension_deficit = (
        (180.0 - peak) / 180.0
    )

    # --------------------------------------------------------
    # Flexion deficit
    # --------------------------------------------------------

    flexion_deficit = (
        np.min(elbow) / 180.0
    )

    features = []

    for i in range(len(seq)):

        t = i / len(seq)

        direction = (
            1.0
            if velocity[i] > 0
            else -1.0
        )

        pause = (
            1.0
            if abs(velocity[i]) < 0.05
            else 0.0
        )

        completion = (
            (elbow[i] - np.min(elbow))
            / rom
        )

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

    features = np.array(
        features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Fixed sequence length
    # --------------------------------------------------------

    if len(features) < SEQUENCE_LENGTH:

        pad = np.zeros(
            (
                SEQUENCE_LENGTH - len(features),
                features.shape[1]
            ),
            dtype=np.float32
        )

        features = np.vstack(
            [features, pad]
        )

    else:

        features = features[
            -SEQUENCE_LENGTH:
        ]

    return features


# ============================================================
# FULL REP DETECTOR
#
# IMPORTANT:
#
# One complete rep is:
#
# DOWN → UP → DOWN
#
# The first valid outbound direction is locked.
# The opposite direction cannot start another rep.
# ============================================================

class FullRepDetector:

    def __init__(self, fps):

        self.fps = fps

        # ----------------------------------------------------
        # Detector state
        # ----------------------------------------------------

        self.state = "READY"

        # ----------------------------------------------------
        # Starting DOWN position
        # ----------------------------------------------------

        self.baseline_angle = None

        # ----------------------------------------------------
        # Direction used for the OUTBOUND movement.
        #
        # This is learned from the first valid rep.
        #
        # Example:
        #
        # UP = elbow angle increasing
        #
        # Then:
        #
        # UP   = start rep
        # DOWN = return
        #
        # ----------------------------------------------------

        self.rep_direction = None

        # ----------------------------------------------------
        # Current movement direction
        # ----------------------------------------------------

        self.direction = None

        # ----------------------------------------------------
        # Extreme angle reached during UP movement
        # ----------------------------------------------------

        self.extreme_angle = None

        # ----------------------------------------------------
        # Frame where rep started
        # ----------------------------------------------------

        self.start_frame = None

        # ----------------------------------------------------
        # Maximum ROM reached
        # ----------------------------------------------------

        self.max_rom = 0.0

        # ----------------------------------------------------
        # Velocity buffer
        # ----------------------------------------------------

        self.velocity_buffer = deque(
            maxlen=VELOCITY_BUFFER_SIZE
        )

        # ----------------------------------------------------
        # Last completed rep
        # ----------------------------------------------------

        self.last_rep_frame = -10000

        self.cooldown_frames = max(
            1,
            int(
                REP_COOLDOWN_SECONDS * fps
            )
        )

        # ----------------------------------------------------
        # Live ROM
        # ----------------------------------------------------

        self.current_rom = 0.0

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, baseline=None):

        self.state = "READY"

        self.baseline_angle = baseline

        self.direction = None

        self.extreme_angle = None

        self.start_frame = None

        self.max_rom = 0.0

        self.current_rom = 0.0

        self.velocity_buffer.clear()

        # IMPORTANT:
        # Do NOT reset rep_direction once learned.
        #
        # If it was UP, we keep it UP.
        # Therefore DOWN can never start another rep.
        #
        # During initial calibration rep_direction is None.

    # ========================================================
    # START NEW REP
    # ========================================================

    def start_rep(
        self,
        angle,
        frame_index
    ):

        self.state = "MOVING"

        self.direction = self.rep_direction

        self.extreme_angle = angle

        self.start_frame = frame_index

        self.max_rom = 0.0

        self.current_rom = 0.0

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        angle,
        velocity,
        frame_index
    ):

        # ----------------------------------------------------
        # Smooth velocity
        # ----------------------------------------------------

        self.velocity_buffer.append(
            velocity
        )

        smooth_velocity = float(
            np.mean(
                self.velocity_buffer
            )
        )

        # ----------------------------------------------------
        # Determine current movement
        # ----------------------------------------------------

        if smooth_velocity > MOVEMENT_THRESHOLD:

            movement = "UP"

        elif smooth_velocity < -MOVEMENT_THRESHOLD:

            movement = "DOWN"

        else:

            movement = "HOLD"

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if (
            frame_index
            - self.last_rep_frame
            < self.cooldown_frames
        ):

            return None, movement

        # ----------------------------------------------------
        # Baseline safety
        # ----------------------------------------------------

        if self.baseline_angle is None:

            self.baseline_angle = angle

        # ====================================================
        # READY
        # ====================================================

        if self.state == "READY":

            distance = abs(
                angle
                - self.baseline_angle
            )

            # ------------------------------------------------
            # LEARN FIRST REP DIRECTION
            # ------------------------------------------------
            #
            # We only learn the direction when the movement
            # is sufficiently far away from the baseline.
            #
            # This prevents small DOWN movements/noise from
            # immediately creating another repetition.
            # ------------------------------------------------

            if self.rep_direction is None:

                if (
                    movement in ("UP", "DOWN")
                    and distance >= BASELINE_TOLERANCE
                ):

                    self.rep_direction = movement

                    self.start_rep(
                        angle,
                        frame_index
                    )

                    return None, movement

            # ------------------------------------------------
            # AFTER DIRECTION IS LOCKED
            #
            # ONLY the outbound direction can START a rep.
            #
            # If rep_direction = UP:
            #
            # DOWN movement while READY is ignored.
            #
            # ------------------------------------------------

            else:

                if movement == self.rep_direction:

                    self.start_rep(
                        angle,
                        frame_index
                    )

            return None, movement

        # ====================================================
        # MOVING
        #
        # We are performing:
        #
        # DOWN → UP
        #
        # ====================================================

        if self.state == "MOVING":

            # ------------------------------------------------
            # Track the highest excursion.
            # ------------------------------------------------

            if self.rep_direction == "UP":

                if angle > self.extreme_angle:

                    self.extreme_angle = angle

            else:

                if angle < self.extreme_angle:

                    self.extreme_angle = angle

            # ------------------------------------------------
            # Calculate current ROM from baseline.
            # ------------------------------------------------

            self.current_rom = abs(
                self.extreme_angle
                - self.baseline_angle
            )

            self.max_rom = max(
                self.max_rom,
                self.current_rom
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # We only move to RETURNING after the movement
            # reverses from the original direction.
            # ------------------------------------------------

            if self.max_rom >= MIN_REP_ROM:

                opposite_direction = (
                    "DOWN"
                    if self.rep_direction == "UP"
                    else "UP"
                )

                if movement == opposite_direction:

                    self.state = "RETURNING"

            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            if self.start_frame is not None:

                duration = (
                    frame_index
                    - self.start_frame
                ) / self.fps

                if duration > MAX_REP_SECONDS:

                    self.reset(
                        baseline=angle
                    )

            return None, movement

        # ====================================================
        # RETURNING
        #
        # We are performing:
        #
        # UP → DOWN
        #
        # ====================================================

        if self.state == "RETURNING":

            # ------------------------------------------------
            # Continue tracking extreme in case the user
            # briefly continues before returning.
            # ------------------------------------------------

            if self.rep_direction == "UP":

                if angle > self.extreme_angle:

                    self.extreme_angle = angle

            else:

                if angle < self.extreme_angle:

                    self.extreme_angle = angle

            self.current_rom = abs(
                self.extreme_angle
                - self.baseline_angle
            )

            self.max_rom = max(
                self.max_rom,
                self.current_rom
            )

            # ------------------------------------------------
            # Distance from DOWN starting position.
            # ------------------------------------------------

            distance_from_baseline = abs(
                angle
                - self.baseline_angle
            )

            # ------------------------------------------------
            # Duration
            # ------------------------------------------------

            duration_seconds = (
                frame_index
                - self.start_frame
            ) / self.fps

            # ------------------------------------------------
            # COMPLETE FULL REP
            #
            # The rep is counted ONLY when:
            #
            # 1. It went in the locked outbound direction
            # 2. It reached minimum ROM
            # 3. It reversed
            # 4. It returned to DOWN
            # ------------------------------------------------

            if (
                distance_from_baseline
                <= BASELINE_TOLERANCE
                and self.max_rom
                >= MIN_REP_ROM
                and duration_seconds
                >= MIN_REP_SECONDS
                and duration_seconds
                <= MAX_REP_SECONDS
            ):

                rep = (
                    self.start_frame,
                    frame_index
                )

                # ------------------------------------------------
                # New DOWN baseline.
                # ------------------------------------------------

                self.baseline_angle = angle

                self.last_rep_frame = frame_index

                # ------------------------------------------------
                # Ready for the NEXT UP movement.
                # ------------------------------------------------

                self.state = "READY"

                self.direction = None

                self.extreme_angle = None

                self.start_frame = None

                self.max_rom = 0.0

                self.current_rom = 0.0

                self.velocity_buffer.clear()

                return rep, movement

            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            if (
                self.start_frame is not None
                and duration_seconds > MAX_REP_SECONDS
            ):

                self.reset(
                    baseline=angle
                )

            return None, movement

        return None, movement


# ============================================================
# MOVEMENT ANALYSIS
# ============================================================

def calculate_shoulder_movement(seq):

    if len(seq) < 2:
        return 0.0

    shoulders = seq[:, 7:9]

    movement = np.linalg.norm(
        shoulders - shoulders[0],
        axis=1
    )

    return float(
        np.max(movement)
    )


def calculate_smoothness(seq):

    if len(seq) < 3:
        return 0.0

    elbow = seq[:, 0]

    velocity = np.gradient(
        elbow
    )

    return float(
        np.std(
            np.diff(velocity)
        )
    )


def calculate_rep_duration(
    seq,
    fps
):

    if fps <= 0:
        return 0.0

    return float(
        len(seq) / fps
    )


# ============================================================
# REP ANALYSIS
# ============================================================

def analyze_rep(
    seq,
    fps
):

    seq = np.array(
        seq,
        dtype=np.float32
    )

    if len(seq) < 5:

        return {
            "rom": 0.0,
            "shoulder_movement": 0.0,
            "smoothness": 999.0,
            "duration": 0.0
        }

    elbow = seq[:, 0]

    rom = float(
        np.ptp(elbow)
    )

    shoulder_movement = (
        calculate_shoulder_movement(
            seq
        )
    )

    smoothness = (
        calculate_smoothness(
            seq
        )
    )

    duration = (
        calculate_rep_duration(
            seq,
            fps
        )
    )

    return {
        "rom": rom,
        "shoulder_movement":
            shoulder_movement,
        "smoothness":
            smoothness,
        "duration":
            duration
    }


# ============================================================
# SCORE FUNCTIONS
# ============================================================

def calculate_rom_score(rom):

    if rom >= ROM_EXCELLENT:
        return 100

    if rom >= ROM_GOOD:
        return 95

    if rom >= ROM_ACCEPTABLE:
        return 80

    if rom >= ROM_MINIMUM:
        return 55

    return 20


def calculate_shoulder_score(
    value
):

    if value <= SHOULDER_GOOD:
        return 100

    if value <= SHOULDER_ACCEPTABLE:
        return 75

    if value <= SHOULDER_BAD:
        return 50

    return 25


def calculate_smoothness_score(
    value
):

    if value <= SMOOTHNESS_GOOD:
        return 100

    if value <= SMOOTHNESS_ACCEPTABLE:
        return 85

    if value <= SMOOTHNESS_BAD:
        return 65

    return 40


# ============================================================
# FEEDBACK + CLASSIFICATION
# ============================================================

def generate_feedback(
    analysis,
    model_probability
):

    rom = analysis["rom"]

    shoulder_movement = (
        analysis["shoulder_movement"]
    )

    smoothness = analysis["smoothness"]

    duration = analysis["duration"]

    feedback = []

    # ========================================================
    # SCORES
    # ========================================================

    rom_score = calculate_rom_score(
        rom
    )

    shoulder_score = (
        calculate_shoulder_score(
            shoulder_movement
        )
    )

    smooth_score = (
        calculate_smoothness_score(
            smoothness
        )
    )

    ai_score = (
        float(model_probability)
        * 100.0
    )

    # ========================================================
    # INDIVIDUAL PROBLEMS
    # ========================================================

    problems = []

    positives = []

    # --------------------------------------------------------
    # ROM
    # --------------------------------------------------------

    if rom < ROM_MINIMUM:

        problems.append(
            (
                "BAD",
                "Insufficient elbow range of motion"
            )
        )

    elif rom < ROM_ACCEPTABLE:

        problems.append(
            (
                "WARN",
                "Elbow range of motion was too small"
            )
        )

    elif rom < ROM_GOOD:

        problems.append(
            (
                "WARN",
                "Try to use a larger elbow range of motion"
            )
        )

    else:

        positives.append(
            (
                "GOOD",
                "Good elbow range of motion"
            )
        )

    # --------------------------------------------------------
    # SHOULDER
    # --------------------------------------------------------

    if shoulder_movement > SHOULDER_BAD:

        problems.append(
            (
                "BAD",
                "Too much shoulder movement was detected"
            )
        )

    elif shoulder_movement > SHOULDER_ACCEPTABLE:

        problems.append(
            (
                "WARN",
                "Keep your shoulder more stable"
            )
        )

    else:

        positives.append(
            (
                "GOOD",
                "Shoulder remained stable"
            )
        )

    # --------------------------------------------------------
    # SMOOTHNESS
    # --------------------------------------------------------

    if smoothness > SMOOTHNESS_BAD:

        problems.append(
            (
                "BAD",
                "Movement was too jerky"
            )
        )

    elif smoothness > SMOOTHNESS_ACCEPTABLE:

        problems.append(
            (
                "WARN",
                "Perform the movement more smoothly"
            )
        )

    else:

        positives.append(
            (
                "GOOD",
                "Movement was controlled"
            )
        )

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    if duration < REP_DURATION_FAST:

        problems.append(
            (
                "WARN",
                "Movement was too fast; slow down"
            )
        )

    elif duration > REP_DURATION_SLOW:

        problems.append(
            (
                "WARN",
                "Movement was too slow; maintain a steady pace"
            )
        )

    else:

        positives.append(
            (
                "GOOD",
                "Good movement pace"
            )
        )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    final_score = (
        rom_score * WEIGHT_ROM
        + shoulder_score * WEIGHT_SHOULDER
        + smooth_score * WEIGHT_SMOOTHNESS
        + ai_score * WEIGHT_AI
    )

    final_score = int(
        np.clip(
            round(final_score),
            0,
            100
        )
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    severe_geometry_failure = (
        rom < ROM_MINIMUM
        or shoulder_movement > SHOULDER_BAD
    )

    if model_probability < AI_REJECT_THRESHOLD:

        label = "INCORRECT"

    elif severe_geometry_failure:

        label = "INCORRECT"

    elif (
        model_probability >= AI_CORRECT_THRESHOLD
        and final_score >= 68
    ):

        label = "CORRECT"

    elif (
        model_probability >= 0.50
        and final_score >= 78
        and rom >= ROM_ACCEPTABLE
        and shoulder_movement
        <= SHOULDER_ACCEPTABLE
    ):

        label = "CORRECT"

    else:

        label = "INCORRECT"

    # ========================================================
    # IMPORTANT:
    #
    # If INCORRECT, feedback must explain WHY.
    # ========================================================

    if label == "INCORRECT":

        feedback = []

        # ----------------------------------------------------
        # Physical problems first
        # ----------------------------------------------------

        for item in problems:

            feedback.append(item)

        # ----------------------------------------------------
        # AI-specific problem
        # ----------------------------------------------------

        if model_probability < AI_REJECT_THRESHOLD:

            feedback.append(
                (
                    "BAD",
                    "AI detected a movement pattern inconsistent with correct form"
                )
            )

        elif model_probability < AI_CORRECT_THRESHOLD:

            feedback.append(
                (
                    "WARN",
                    "AI confidence for correct form was low"
                )
            )

        # ----------------------------------------------------
        # If no obvious physical issue exists,
        # explain that the AI was the deciding factor.
        # ----------------------------------------------------

        if len(feedback) == 0:

            feedback.append(
                (
                    "BAD",
                    "Movement pattern did not match the learned correct form"
                )
            )

            feedback.append(
                (
                    "WARN",
                    "Repeat the movement slowly with controlled form"
                )
            )

        # ----------------------------------------------------
        # Limit to the most useful 3 messages.
        # ----------------------------------------------------

        feedback = feedback[:3]

    # ========================================================
    # CORRECT REP FEEDBACK
    # ========================================================

    else:

        feedback = []

        # ----------------------------------------------------
        # Don't overload correct reps with warnings.
        # ----------------------------------------------------

        if positives:

            feedback.extend(
                positives[:3]
            )

        if not feedback:

            feedback.append(
                (
                    "GOOD",
                    "Good repetition"
                )
            )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_feedback = []

    seen = set()

    for item in feedback:

        message = item[1]

        if message not in seen:

            unique_feedback.append(
                item
            )

            seen.add(
                message
            )

    return {
        "label": label,
        "score": final_score,
        "feedback": unique_feedback,
        "model_probability":
            model_probability,
        "rom": rom,
        "shoulder_movement":
            shoulder_movement,
        "smoothness":
            smoothness,
        "duration":
            duration
    }


# ============================================================
# BODY WIREFRAME
# ============================================================

def draw_wireframe(
    frame,
    results
):

    if (
        results is None
        or results.pose_landmarks is None
    ):

        return

    landmarks = (
        results.pose_landmarks.landmark
    )

    connections = [

        (11, 12),

        (12, 14),
        (14, 16),

        (11, 13),
        (13, 15),

        (11, 23),
        (12, 24),
        (23, 24),

        (24, 26),
        (26, 28),

        (23, 25),
        (25, 27)
    ]

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Lines
    # --------------------------------------------------------

    for a, b in connections:

        if (
            landmarks[a].visibility
            < MIN_VISIBILITY
            or landmarks[b].visibility
            < MIN_VISIBILITY
        ):

            continue

        x1 = int(
            landmarks[a].x * w
        )

        y1 = int(
            landmarks[a].y * h
        )

        x2 = int(
            landmarks[b].x * w
        )

        y2 = int(
            landmarks[b].y * h
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            CYAN,
            3
        )

    # --------------------------------------------------------
    # Joints
    # --------------------------------------------------------

    important = [
        11, 12,
        13, 14,
        15, 16,
        23, 24,
        25, 26,
        27, 28
    ]

    for idx in important:

        if (
            landmarks[idx].visibility
            < MIN_VISIBILITY
        ):

            continue

        x = int(
            landmarks[idx].x * w
        )

        y = int(
            landmarks[idx].y * h
        )

        cv2.circle(
            frame,
            (x, y),
            6,
            WHITE,
            -1
        )

        cv2.circle(
            frame,
            (x, y),
            8,
            BLUE,
            2
        )


# ============================================================
# UI
# ============================================================

def draw_panel(
    frame,
    rep_count,
    movement,
    angle,
    last_result,
    detector_state,
    detector_rom,
    calibrated,
    visibility_ok,
    rep_direction
):

    h, w = frame.shape[:2]

    panel_x1 = 15
    panel_y1 = 15

    panel_x2 = min(
        w - 15,
        600
    )

    panel_y2 = min(
        h - 15,
        560
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        BLACK,
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "HEMOPHYSIO AI",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        CYAN,
        2
    )

    # --------------------------------------------------------
    # REP COUNT
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Reps: {rep_count}",
        (30, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        GREEN,
        2
    )

    # --------------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------------

    if movement == "UP":

        movement_color = GREEN

    elif movement == "DOWN":

        movement_color = YELLOW

    else:

        movement_color = WHITE

    cv2.putText(
        frame,
        f"Motion: {movement}",
        (220, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        movement_color,
        2
    )

    # --------------------------------------------------------
    # ANGLE
    # --------------------------------------------------------

    if angle is not None:

        angle_text = (
            f"Angle: {angle:.1f} deg"
        )

    else:

        angle_text = "Angle: --"

    cv2.putText(
        frame,
        angle_text,
        (30, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        WHITE,
        2
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    if not calibrated:

        cv2.putText(
            frame,
            "CALIBRATING...",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            YELLOW,
            2
        )

        return

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    if visibility_ok:

        tracking_color = GREEN

        tracking_text = (
            "Tracking: GOOD"
        )

    else:

        tracking_color = RED

        tracking_text = (
            "Tracking: CHECK POSITION"
        )

    cv2.putText(
        frame,
        tracking_text,
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        tracking_color,
        2
    )

    # --------------------------------------------------------
    # REP DIRECTION
    # --------------------------------------------------------

    if rep_direction is None:

        direction_text = (
            "Rep direction: Learning..."
        )

    else:

        direction_text = (
            f"Rep cycle: {rep_direction} -> "
            f"{'DOWN' if rep_direction == 'UP' else 'UP'}"
        )

    cv2.putText(
        frame,
        direction_text,
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        CYAN,
        1
    )

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Detector: {detector_state}",
        (330, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1
    )

    # --------------------------------------------------------
    # LIVE ROM
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Live ROM: {detector_rom:.1f} deg",
        (30, 208),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1
    )

    # --------------------------------------------------------
    # NO RESULT
    # --------------------------------------------------------

    if last_result is None:

        cv2.putText(
            frame,
            "Complete DOWN -> UP -> DOWN",
            (30, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            GRAY,
            2
        )

        return

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status = last_result["label"]

    status_color = (
        GREEN
        if status == "CORRECT"
        else RED
    )

    cv2.putText(
        frame,
        f"Last Rep: {status}",
        (30, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Score: {last_result['score']}/100",
        (30, 278),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        WHITE,
        2
    )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"AI: {last_result['model_probability'] * 100:.1f}%",
        (300, 278),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        2
    )

    # --------------------------------------------------------
    # ROM
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"ROM: {last_result['rom']:.1f} deg",
        (30, 308),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        1
    )

    # --------------------------------------------------------
    # SHOULDER
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Shoulder: {last_result['shoulder_movement']:.3f}",
        (300, 308),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        WHITE,
        1
    )

    # --------------------------------------------------------
    # FEEDBACK HEADER
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Feedback:",
        (30, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        CYAN,
        2
    )

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    y = 370

    for feedback_type, message in (
        last_result["feedback"]
    ):

        if feedback_type == "GOOD":

            color = GREEN
            prefix = "+"

        elif feedback_type == "WARN":

            color = YELLOW
            prefix = "!"

        else:

            color = RED
            prefix = "X"

        if len(message) > 65:

            message = (
                message[:62]
                + "..."
            )

        cv2.putText(
            frame,
            f"{prefix} {message}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1
        )

        y += 25

        if y > panel_y2 - 15:

            break


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = ElbowLSTM(
        input_size=22
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# LOAD NORMALIZATION
# ============================================================

def load_normalization():

    if not MEAN_PATH.exists():

        raise FileNotFoundError(
            f"Missing normalization file: {MEAN_PATH}"
        )

    if not STD_PATH.exists():

        raise FileNotFoundError(
            f"Missing normalization file: {STD_PATH}"
        )

    mean = np.load(
        MEAN_PATH
    )

    std = np.load(
        STD_PATH
    )

    return mean, std


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 65)
    print("              HEMOPHYSIO REALTIME AI")
    print("=" * 65)
    print()

    # ========================================================
    # MODEL
    # ========================================================

    print("🔄 Loading model...")

    model = load_model()

    print("✅ Model loaded")

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print("🔄 Loading normalization...")

    mean, std = load_normalization()

    print("✅ Normalization loaded")

    # ========================================================
    # CAMERA
    # ========================================================

    print("📷 Starting camera...")

    cam = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cam.isOpened():

        print("❌ Camera not working")

        return

    # ========================================================
    # FPS
    # ========================================================

    fps = cam.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    print(
        f"🎥 Camera FPS: {fps:.1f}"
    )

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ========================================================
    # VARIABLES
    # ========================================================

    prev_angle = None

    prev_ema = None

    all_data = []

    calibration_frames = 0

    calibration_required = int(
        fps * CALIBRATION_SECONDS
    )

    calibrated = False

    detector = FullRepDetector(
        fps
    )

    rep_count = 0

    movement = "IDLE"

    current_angle = None

    last_result = None

    visibility_ok = True

    # ========================================================
    # START
    # ========================================================

    print()

    print(
        "🟡 Calibration started..."
    )

    print(
        "Keep your arm in the DOWN starting position."
    )

    print(
        "The first complete cycle will be:"
    )

    print(
        "DOWN → UP → DOWN"
    )

    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cam.read()

        if not ret:

            print(
                "❌ Could not read camera frame"
            )

            break

        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Pose extraction
        # ----------------------------------------------------

        row, results = (
            extract_frame_landmarks(
                frame,
                pose
            )
        )

        # ----------------------------------------------------
        # Wireframe
        # ----------------------------------------------------

        if results is not None:

            draw_wireframe(
                frame,
                results
            )

        # ====================================================
        # LANDMARK AVAILABLE
        # ====================================================

        if row is not None:

            # ------------------------------------------------
            # Visibility
            # ------------------------------------------------

            visibility_ok = (
                row["shoulder_visibility"]
                >= MIN_VISIBILITY

                and

                row["elbow_visibility"]
                >= MIN_VISIBILITY

                and

                row["wrist_visibility"]
                >= MIN_VISIBILITY
            )

            if not visibility_ok:

                movement = "CHECK"

            # ------------------------------------------------
            # Extract angles
            # ------------------------------------------------

            data = extract_angles(
                row
            )

            raw_angle = data[
                "elbow_angle"
            ]

            # ------------------------------------------------
            # Smooth angle
            # ------------------------------------------------

            current_angle, prev_ema = (
                smooth_angle(
                    raw_angle,
                    prev_ema
                )
            )

            # ------------------------------------------------
            # Frame data
            #
            # SAME ORDER AS TRAINING
            # ------------------------------------------------

            frame_data = [

                data["elbow_angle"],

                data["shoulder_angle"],

                data["wrist_angle"],

                *data["elbow"],

                *data["wrist"],

                *data["shoulder"],

                data["arm_length"]
            ]

            all_data.append(
                frame_data
            )

            # ------------------------------------------------
            # Velocity
            # ------------------------------------------------

            if prev_angle is None:

                velocity = 0.0

            else:

                velocity = (
                    current_angle
                    - prev_angle
                )

            prev_angle = current_angle

            # =================================================
            # CALIBRATION
            # =================================================

            if not calibrated:

                calibration_frames += 1

                if (
                    calibration_frames
                    >= calibration_required
                ):

                    calibrated = True

                    detector.reset(
                        baseline=current_angle
                    )

                    print()

                    print(
                        "🟢 Calibration complete."
                    )

                    print(
                        f"📐 DOWN baseline angle: "
                        f"{current_angle:.1f}°"
                    )

                    print(
                        "▶ Raise your arm UP, then bring it DOWN."
                    )

                    print(
                        "▶ Only DOWN → UP → DOWN counts as 1 rep."
                    )

                    print()

            # =================================================
            # REP DETECTION
            # =================================================

            else:

                if visibility_ok:

                    rep, movement = (
                        detector.update(
                            current_angle,
                            velocity,
                            len(all_data) - 1
                        )
                    )

                    # ==========================================
                    # FULL REP COMPLETED
                    # ==========================================

                    if rep is not None:

                        start_frame, end_frame = rep

                        # ------------------------------------------------
                        # Safety checks
                        # ------------------------------------------------

                        if (
                            start_frame < 0
                            or end_frame
                            >= len(all_data)
                            or end_frame
                            <= start_frame
                        ):

                            continue

                        # ------------------------------------------------
                        # Exact complete cycle
                        # ------------------------------------------------

                        seq = all_data[
                            start_frame:
                            end_frame + 1
                        ]

                        # ------------------------------------------------
                        # Analyze movement
                        # ------------------------------------------------

                        analysis = analyze_rep(
                            seq,
                            fps
                        )

                        # ------------------------------------------------
                        # Build model features
                        # ------------------------------------------------

                        features = build_features(
                            seq
                        )

                        if features is None:

                            continue

                        # ------------------------------------------------
                        # Normalize
                        # ------------------------------------------------

                        features = (
                            features - mean
                        ) / std

                        # ------------------------------------------------
                        # Tensor
                        # ------------------------------------------------

                        input_tensor = (
                            torch.from_numpy(
                                features
                            ).float()
                        )

                        if input_tensor.ndim == 2:

                            input_tensor = (
                                input_tensor
                                .unsqueeze(0)
                            )

                        # ------------------------------------------------
                        # Shape check
                        # ------------------------------------------------

                        if (
                            input_tensor.shape[1]
                            != SEQUENCE_LENGTH
                        ):

                            print(
                                "⚠️ Bad sequence shape:",
                                input_tensor.shape
                            )

                            continue

                        if (
                            input_tensor.shape[2]
                            != 22
                        ):

                            print(
                                "⚠️ Bad feature shape:",
                                input_tensor.shape
                            )

                            continue

                        # ------------------------------------------------
                        # AI prediction
                        # ------------------------------------------------

                        with torch.no_grad():

                            logits = model(
                                input_tensor
                            )

                            probability = (
                                torch.sigmoid(
                                    logits
                                ).item()
                            )

                        # ------------------------------------------------
                        # Final result
                        # ------------------------------------------------

                        last_result = (
                            generate_feedback(
                                analysis,
                                probability
                            )
                        )

                        # ------------------------------------------------
                        # Count ONLY completed full reps
                        # ------------------------------------------------

                        rep_count += 1

                        # =================================================
                        # CONSOLE OUTPUT
                        # =================================================

                        print()

                        print("=" * 65)

                        print(
                            f"✅ REP {rep_count} COMPLETED"
                        )

                        print(
                            "Cycle: DOWN → UP → DOWN"
                        )

                        print(
                            f"AI confidence: "
                            f"{probability * 100:.1f}%"
                        )

                        print(
                            f"ROM: "
                            f"{analysis['rom']:.1f}°"
                        )

                        print(
                            f"Shoulder movement: "
                            f"{analysis['shoulder_movement']:.3f}"
                        )

                        print(
                            f"Smoothness: "
                            f"{analysis['smoothness']:.2f}"
                        )

                        print(
                            f"Duration: "
                            f"{analysis['duration']:.2f}s"
                        )

                        print(
                            f"Score: "
                            f"{last_result['score']}/100"
                        )

                        print(
                            f"Status: "
                            f"{last_result['label']}"
                        )

                        print(
                            "Feedback:"
                        )

                        for (
                            feedback_type,
                            message
                        ) in last_result[
                            "feedback"
                        ]:

                            print(
                                f"  • {message}"
                            )

                        print(
                            "🔄 Ready for next "
                            "UP → DOWN cycle"
                        )

                        print("=" * 65)

        # ====================================================
        # NO LANDMARK
        # ====================================================

        else:

            visibility_ok = False

        # ====================================================
        # UI
        # ====================================================

        draw_panel(
            frame,
            rep_count,
            movement,
            current_angle,
            last_result,
            detector.state,
            detector.current_rom,
            calibrated,
            visibility_ok,
            detector.rep_direction
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Elbow AI",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ----------------------------------------------------
        # ESC
        # ----------------------------------------------------

        if key == 27:

            break

        # ----------------------------------------------------
        # Q
        # ----------------------------------------------------

        if key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cam.release()

    pose.close()

    cv2.destroyAllWindows()

    print()

    print("=" * 65)

    print(
        f"Session completed | "
        f"Total complete reps: {rep_count}"
    )

    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run()
import numpy as np


def smooth_signal(x, window=5):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="same")


def find_peaks(signal, min_distance=5):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return peaks


def find_valleys(signal, min_distance=5):
    valleys = []
    for i in range(1, len(signal)-1):
        if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            if not valleys or (i - valleys[-1]) >= min_distance:
                valleys.append(i)
    return valleys


def detect_rep_boundaries(
    angles,
    fps=30.0,
):
    angles = np.array(angles)

    if len(angles) < 10:
        return []

    # --------------------------
    # SMOOTH SIGNAL
    # --------------------------
    smooth_angles_arr = smooth_signal(angles, window=5)

    # --------------------------
    # RANGE CHECK
    # --------------------------
    rom = np.max(smooth_angles_arr) - np.min(smooth_angles_arr)

    if rom < 8:
        return []

    # --------------------------
    # FIND PEAKS & VALLEYS
    # --------------------------
    peaks = find_peaks(smooth_angles_arr, min_distance=5)
    valleys = find_valleys(smooth_angles_arr, min_distance=5)

    print(f"Peaks: {len(peaks)}, Valleys: {len(valleys)}")  # DEBUG

    if not peaks or not valleys:
        return []

    reps = []

    # --------------------------
    # valley → peak → valley
    # --------------------------
    for peak in peaks:

        prev_valleys = [v for v in valleys if v < peak]
        next_valleys = [v for v in valleys if v > peak]

        if not prev_valleys or not next_valleys:
            continue

        start = prev_valleys[-1]
        end = next_valleys[0]

        duration = end - start
        min_frames = max(5, int(0.15 * fps))

        if duration < min_frames:
            continue

        reps.append((start, end))

    return reps
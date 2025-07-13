"""
src/filters.py

Irrelevant-frame filters for video cleanup:
1. is_corrupted(frame)
2. is_blank_or_white(frame, epsilon_black, epsilon_white)
3. is_blurry_or_noisy(frame, threshold_blur, threshold_noise)
4. is_near_duplicate(frame, last_frame, threshold_diff)  
"""

import cv2
import numpy as np

# Static thresholds (tune these after calibration)
EPSILON_BLACK = 5        # mean intensity ≤ this ⇒ nearly black
EPSILON_WHITE = 245      # mean intensity ≥ this ⇒ nearly white / overexposed
THRESHOLD_BLUR = 70.0   # var(Laplacian) ≤ this ⇒ blurry
THRESHOLD_NOISE = 45.0   # noise_metric ≥ this ⇒ noisy
THRESHOLD_DIFF = 0.8    # mean absolute pixel diff ≤ this ⇒ near-duplicate
THRESHOLD_SPIKE = 25.0  # Δ mean intensity ≥ 34 → brightness-spike frames
THRESHOLD_HIST = 0.4  # hist correlation ≤ 0.72 → histogram-outlier frames


def is_corrupted(frame: np.ndarray) -> bool:
    """
    Returns True if frame is None, empty, or contains NaNs.
    """
    if frame is None or frame.size == 0:
        return True
    if np.isnan(frame).any():
        return True
    return False


def is_blank_or_white(frame: np.ndarray,
                      epsilon_black: float = EPSILON_BLACK,
                      epsilon_white: float = EPSILON_WHITE) -> bool:
    """
    Returns True if mean intensity ≤ epsilon_black (black) or
    mean intensity ≥ epsilon_white (white/overexposed).
    """
    mean_intensity = float(np.mean(frame))
    return (mean_intensity <= epsilon_black) or (mean_intensity >= epsilon_white)


def is_blurry_or_noisy(frame: np.ndarray,
                       threshold_blur: float = THRESHOLD_BLUR,
                       threshold_noise: float = THRESHOLD_NOISE) -> bool:
    """
    Returns True if frame is either too blurry or too noisy.
    - Blurry: variance of Laplacian ≤ threshold_blur
    - Noisy: mean squared diff from median ≥ threshold_noise
    """
    # Blur detection
    lap = cv2.Laplacian(frame, cv2.CV_64F)
    var_lap = float(lap.var())
    if var_lap <= threshold_blur:
        return True

    # Noise detection
    median = cv2.medianBlur(frame, 3)
    noise_metric = float(
        np.mean((frame.astype(np.float32) - median.astype(np.float32))**2))
    if noise_metric >= threshold_noise:
        return True

    return False


def is_near_duplicate(frame: np.ndarray,
                      last_frame: np.ndarray,
                      threshold_diff: float = THRESHOLD_DIFF) -> bool:
    """
    Returns True if the mean absolute difference between current frame and the
    last kept frame is less than or equal to threshold_diff (i.e., nearly identical).
    """
    if last_frame is None:
        return False
    # Compute mean absolute pixel difference
    diff = np.abs(frame.astype(np.float32) - last_frame.astype(np.float32))
    mean_diff = float(np.mean(diff))
    return mean_diff <= threshold_diff


def is_brightness_spike(gray: np.ndarray,
                        last_gray: np.ndarray,
                        threshold_spike: float = THRESHOLD_SPIKE) -> bool:
    """
    Drop frames where the mean intensity jumps by more than threshold_spike
    compared to the last kept frame (camera glitches / exposure flicker).
    """
    if last_gray is None:
        return False
    curr_mean = float(np.mean(gray))
    prev_mean = float(np.mean(last_gray))
    return abs(curr_mean - prev_mean) >= threshold_spike


def is_histogram_outlier(frame: np.ndarray,
                         last_frame: np.ndarray,
                         threshold_hist: float = THRESHOLD_HIST) -> bool:
    """
    Drop frames whose grayscale histogram correlation with the last kept
    frame is below threshold_hist (i.e. they look unlike the normal video).
    """
    if last_frame is None:
        return False
    # Compute 256‐bin histograms
    hist1 = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([last_frame], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # Drop if correlation is too low
    return corr < threshold_hist

import os
import cv2
import csv
import numpy as np
from pathlib import Path
import filters  # filters.py in the same directory

# Directories and paths
PROCESSED_DIR = Path("data/processed")
CLEANED_DIR = Path("results/cleaned_videos")
LOG_CSV = Path("logs") / "frame_by_frame.csv"


def compute_metrics(gray_frame: np.ndarray, last_gray: np.ndarray):
    """
    Compute metrics for logging on a grayscale frame:
    - mean_intensity: average pixel value
    - var_lap: variance of Laplacian (blur metric)
    - noise_metric: mean squared difference from median (noise)
    - hist_corr: histogram correlation with last kept frame
    - mean_diff: mean absolute pixel difference vs. last kept frame
    - delta_mean: absolute change in mean intensity vs. last kept frame
    """
    mean_intensity = float(np.mean(gray_frame))

    # Blur metric
    lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
    var_lap = float(lap.var())

    # Noise metric
    median = cv2.medianBlur(gray_frame, 3)
    noise_metric = float(
        np.mean((gray_frame.astype(np.float32) - median.astype(np.float32))**2))

    # Histogram correlation
    hist_corr = None
    if last_gray is not None:
        hist1 = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([last_gray], [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        hist_corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    # Mean absolute pixel difference
    if last_gray is not None:
        diff = np.abs(gray_frame.astype(np.float32) -
                      last_gray.astype(np.float32))
        mean_diff = float(np.mean(diff))
        delta_mean = abs(mean_intensity - float(np.mean(last_gray)))
    else:
        mean_diff = float("nan")
        delta_mean = float("nan")

    return mean_intensity, var_lap, noise_metric, hist_corr, mean_diff, delta_mean


def main():
    # Prepare output directories
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "video", "frame_idx",
            "mean_intensity", "var_lap", "noise_metric", "hist_corr",
            "mean_diff", "delta_mean", "filter_dropped"
        ])

        # Process each video
        for video_path in sorted(PROCESSED_DIR.glob("*.mp4")):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: cannot open {video_path}")
                continue

            # Video writer setup (color output)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = CLEANED_DIR / video_path.name
            out = cv2.VideoWriter(str(out_path), fourcc,
                                  fps, (width, height), isColor=True)

            last_kept_gray = None
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale for filtering and metrics
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Compute all metrics including diff and delta_mean
                mean_int, var_l, noise_m, hist_c, mean_diff, delta_mean = compute_metrics(
                    gray, last_kept_gray)

                # Determine which filter (if any) drops the frame
                dropped_by = None
                if filters.is_corrupted(frame):
                    dropped_by = "corrupted"
                elif filters.is_blank_or_white(gray, filters.EPSILON_BLACK, filters.EPSILON_WHITE):
                    dropped_by = "blank_or_white"
                elif filters.is_blurry_or_noisy(gray, filters.THRESHOLD_BLUR, filters.THRESHOLD_NOISE):
                    dropped_by = "blurry_or_noisy"
                elif filters.is_near_duplicate(gray, last_kept_gray, filters.THRESHOLD_DIFF):
                    dropped_by = "near_duplicate"
                elif filters.is_brightness_spike(gray, last_kept_gray, filters.THRESHOLD_SPIKE):
                    dropped_by = "brightness_spike"
                elif filters.is_histogram_outlier(gray, last_kept_gray, filters.THRESHOLD_HIST):
                    dropped_by = "histogram_outlier"

                # Log metrics and decision
                writer.writerow([
                    video_path.name,
                    frame_idx,
                    f"{mean_int:.2f}",
                    f"{var_l:.2f}",
                    f"{noise_m:.2f}",
                    f"{hist_c:.2f}" if hist_c is not None else "",
                    f"{mean_diff:.2f}",
                    f"{delta_mean:.2f}",
                    dropped_by or "kept"
                ])

                # Write the frame if not dropped
                if dropped_by is None:
                    out.write(frame)
                    last_kept_gray = gray.copy()

                frame_idx += 1

            cap.release()
            out.release()
            print(f"Cleaned: {video_path.name} → {out_path.name}")

    print(f"Done. Log at {LOG_CSV}")


if __name__ == "__main__":
    main()

"""
src/preprocess.py

Preprocess videos:
- Downsample 60 FPS → 30 FPS
- Resize to 720 px width (maintaining aspect ratio)
- Convert to grayscale
- Strip audio
- Write processed videos and log stats

Usage:
    python src/preprocess.py
"""

import os
import cv2
import csv
from pathlib import Path

# Configuration
TARGET_FPS = 30
TARGET_WIDTH = 720
RAW_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")
LOG_CSV = Path("logs") / "preprocessing_summary.csv"

def preprocess_video(input_path: Path, output_path: Path):
    """
    Preprocess a single video:
    - Downsample to TARGET_FPS
    - Resize to TARGET_WIDTH while preserving aspect ratio
    - Convert to grayscale
    - Save to output_path
    Returns: tuple of (orig_fps, orig_frames, new_frames, orig_res, new_res)
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")
    # Original properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate downsample interval
    frame_interval = max(1, int(round(orig_fps / TARGET_FPS)))

    # Calculate target height to maintain aspect ratio
    scale = TARGET_WIDTH / orig_width
    target_height = int(orig_height * scale)

    # Prepare VideoWriter (grayscale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, TARGET_FPS, (TARGET_WIDTH, target_height), isColor=False)

    new_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Downsample
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        # Resize
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        # Write frame
        out.write(gray)
        new_frames += 1
        frame_idx += 1

    cap.release()
    out.release()
    return orig_fps, orig_frames, new_frames, (orig_width, orig_height), (TARGET_WIDTH, target_height)

def main():
    # Ensure directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Open CSV log
    with open(LOG_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "video", "orig_fps", "orig_frame_count", "new_frame_count",
            "orig_width", "orig_height", "new_width", "new_height"
        ])

        # Process each .mp4 in RAW_DIR
        for video_path in sorted(RAW_DIR.glob("*.mp4")):
            output_path = PROCESSED_DIR / video_path.name
            print(f"Processing {video_path.name} → {output_path.name}")
            orig_fps, orig_fc, new_fc, orig_res, new_res = preprocess_video(video_path, output_path)
            writer.writerow([
                video_path.name, orig_fps, orig_fc, new_fc,
                orig_res[0], orig_res[1], new_res[0], new_res[1]
            ])

    print(f"Preprocessing complete. Summary written to {LOG_CSV}")

if __name__ == "__main__":
    main()

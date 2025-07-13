"""
src/sample_and_label_frames.py

Balanced and uniform frame sampling + interactive labeling for robust ground truth.

For each video:
  - Uniformly sample N_uniform frames across the timeline.
  - Sample N_dropped frames that the current pipeline dropped.
  - Sample N_kept frames that the pipeline kept.
  - Shuffle and present for labeling ('d' to drop, 'k' to keep, 'q' to quit).

Outputs:
  logs/gt_labels.csv with columns [video, frame_idx, gt_drop].

Usage:
    cd <project_root>
    python src/sample_and_label_frames.py --uniform 300 --dropped 200 --kept 200
"""

import cv2
import pandas as pd
import random
import csv
import numpy as np
from pathlib import Path
import argparse


def sample_frames(df_log, total_frames, N_uniform, N_dropped, N_kept):
    """
    Return a shuffled list of frame indices to label:
      - Uniformly sampled across the video timeline.
      - Randomly sampled from dropped frames.
      - Randomly sampled from kept frames.
    """
    # Uniform sampling across timeline
    uniform_idxs = np.linspace(
        0, total_frames - 1, N_uniform, dtype=int).tolist()

    # Sample from dropped frames
    dropped = df_log[df_log['filter_dropped'] != 'kept']['frame_idx'].tolist()
    dropped_idxs = random.sample(dropped, min(
        len(dropped), N_dropped)) if dropped else []

    # Sample from kept frames
    kept = df_log[df_log['filter_dropped'] == 'kept']['frame_idx'].tolist()
    kept_idxs = random.sample(kept, min(len(kept), N_kept)) if kept else []

    # Combine and shuffle, removing duplicates
    combined = list(set(uniform_idxs + dropped_idxs + kept_idxs))
    random.shuffle(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Sample frames for manual ground-truth labeling")
    parser.add_argument("--uniform", type=int, default=300,
                        help="Number of uniformly sampled frames per video")
    parser.add_argument("--dropped", type=int, default=200,
                        help="Number of sampled dropped frames per video")
    parser.add_argument("--kept", type=int, default=200,
                        help="Number of sampled kept frames per video")
    args = parser.parse_args()

    LOG_CSV = Path("logs") / "frame_by_frame.csv"
    RAW_DIR = Path("data")
    GT_CSV = Path("logs") / "gt_labels.csv"

    # Load per-frame filter log
    df = pd.read_csv(LOG_CSV)
    labels = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for video in df['video'].unique():
        print(
            f"Sampling frames for {video}: uniform={args.uniform}, dropped={args.dropped}, kept={args.kept}")
        cap = cv2.VideoCapture(str(RAW_DIR / video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Determine which frames to label
        df_sub = df[df['video'] == video]
        frames_to_label = sample_frames(df_sub, total_frames,
                                        args.uniform, args.dropped, args.kept)

        for idx in frames_to_label:
            cap = cv2.VideoCapture(str(RAW_DIR / video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue

            # Overlay info and prompt
            info = frame.copy()
            cv2.putText(info, f"{video} - Frame {idx}",
                        (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(info, "d=drop  k=keep  q=quit",
                        (10, 70), font, 0.7, (0, 255, 0), 2)
            cv2.imshow("Label Frame", info)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord('q'):
                break
            elif key == ord('d'):
                gt = 1
            elif key == ord('k'):
                gt = 0
            else:
                continue

            labels.append((video, idx, gt))

    # Save labels
    GT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(GT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame_idx", "gt_drop"])
        writer.writerows(labels)

    print(f"Saved {len(labels)} labels to {GT_CSV}")


if __name__ == "__main__":
    main()

"""
src/false_positive_analysis.py

Analyze which filters are generating the most false positives
(i.e. dropping frames that should be kept), excluding parking_view.mp4.

Usage:
    python src/false_positive_analysis.py
"""

import pandas as pd
from pathlib import Path


def main():
    gt_csv = Path("logs") / "gt_labels.csv"
    log_csv = Path("logs") / "frame_by_frame.csv"

    # Load ground truth and filter log
    df_gt = pd.read_csv(gt_csv)   # video,frame_idx,gt_drop (1=drop,0=keep)
    df_log = pd.read_csv(log_csv)  # includes filter_dropped
    df_log['pred_drop'] = (df_log['filter_dropped'] != 'kept').astype(int)

    # Merge on video and frame_idx
    df = pd.merge(
        df_gt,
        df_log[['video', 'frame_idx', 'filter_dropped', 'pred_drop']],
        on=['video', 'frame_idx'], how='inner'
    )

    # False positives: gt=0 (keep) but pred_drop=1
    fp = df[(df.gt_drop == 0) & (df.pred_drop == 1)]

    # Count by filter
    counts = fp['filter_dropped'].value_counts()
    print("\n=== False Positives by Filter (excluding parking_view.mp4) ===")
    print(counts.to_string())

    # Save breakdown
    out_csv = Path("results") / "false_positive_breakdown.csv"
    counts.to_csv(out_csv, header=['FP_count'])
    print(f"\nSaved breakdown to {out_csv}")


if __name__ == "__main__":
    main()

"""
src/evaluate.py

Evaluate the frame-level classification performance of the cleaning pipeline,
Reads in:
- logs/frame_by_frame.csv (predicted drop/keep decisions)
- logs/gt_labels.csv (ground-truth labels: 1=drop, 0=keep)

Computes for each video:
- TP, FP, FN, TN
- Precision, Recall, F1-score

Saves results to results/evaluation_metrics.csv and prints to console.

Usage:
    python src/evaluate.py
"""

import pandas as pd
from pathlib import Path


def compute_confusion_metrics(df):
    """
    Given a DataFrame with columns [video, frame_idx, gt_drop, pred_drop],
    compute TP, FP, FN, TN, precision, recall, f1 for each video and overall.
    """
    results = []
    # Per-video
    for video, group in df.groupby('video'):
        tp = ((group['gt_drop'] == 1) & (group['pred_drop'] == 1)).sum()
        fp = ((group['gt_drop'] == 0) & (group['pred_drop'] == 1)).sum()
        fn = ((group['gt_drop'] == 1) & (group['pred_drop'] == 0)).sum()
        tn = ((group['gt_drop'] == 0) & (group['pred_drop'] == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results.append({
            'video': video,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision, 'Recall': recall, 'F1-score': f1
        })

    # Overall
    tp = ((df['gt_drop'] == 1) & (df['pred_drop'] == 1)).sum()
    fp = ((df['gt_drop'] == 0) & (df['pred_drop'] == 1)).sum()
    fn = ((df['gt_drop'] == 1) & (df['pred_drop'] == 0)).sum()
    tn = ((df['gt_drop'] == 0) & (df['pred_drop'] == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    results.append({
        'video': 'Overall',
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': precision, 'Recall': recall, 'F1-score': f1
    })

    return pd.DataFrame(results)


def main():
    # Paths
    log_csv = Path("logs") / "frame_by_frame.csv"
    gt_csv = Path("logs") / "gt_labels.csv"
    out_csv = Path("results") / "evaluation_metrics.csv"

    # Load data
    df_log = pd.read_csv(log_csv)
    df_gt = pd.read_csv(gt_csv)

    # Prepare predictions: pred_drop = 1 if filter_dropped != 'kept'
    df_log['pred_drop'] = (df_log['filter_dropped'] != 'kept').astype(int)

    # Merge on video and frame_idx
    df = pd.merge(
        df_gt,
        df_log[['video', 'frame_idx', 'pred_drop']],
        on=['video', 'frame_idx'], how='inner'
    )

    # Compute metrics
    df_metrics = compute_confusion_metrics(df)

    # Print and save
    print("\n=== Frame-Level Evaluation Metrics ===")
    print(df_metrics.to_string(index=False))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_csv, index=False)
    print(f"\nSaved evaluation metrics to {out_csv}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from pathlib import Path
import itertools

"""
src/grid_optimize_thresholds.py

Exhaustive grid search over filter thresholds plus consensus parameter 
(minimum number of filters that must fire) to maximize overall F1-score
on manually labeled frames (logs/gt_labels.csv).

Usage:
    python src/grid_optimize_thresholds.py
"""

# Define search space for each parameter
SEARCH_SPACE = {
    'THRESHOLD_BLUR':   [50, 75, 100, 125, 150, 175, 200],
    'THRESHOLD_NOISE':  [10, 20, 30, 40, 50, 60],
    'THRESHOLD_DIFF':   [0.0, 0.5, 1.0, 1.5, 2.0],
    'THRESHOLD_SPIKE':  [10, 20, 30, 40, 50, 60, 70],
    'THRESHOLD_HIST':   [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'CONSENSUS_K':      [1, 2, 3]  # number of flags required to drop
}

# Fixed black/white thresholds
EPSILON_BLACK = 5.0
EPSILON_WHITE = 245.0


def load_data():
    df_gt = pd.read_csv(Path("logs") / "gt_labels.csv")
    df_log = pd.read_csv(Path("logs") / "frame_by_frame.csv")
    # Merge GT and log on video/frame_idx
    df = pd.merge(df_gt, df_log, on=['video', 'frame_idx'], how='inner')
    return df


def compute_f1_for_config(df, config):
    # Compute boolean flags per filter
    conds = []
    conds.append(df['mean_intensity'] <= EPSILON_BLACK)
    conds.append(df['mean_intensity'] >= EPSILON_WHITE)
    conds.append(df['var_lap'] <= config['THRESHOLD_BLUR'])
    conds.append(df['noise_metric'] >= config['THRESHOLD_NOISE'])
    conds.append(df['mean_diff'] <= config['THRESHOLD_DIFF'])
    conds.append(df['delta_mean'] >= config['THRESHOLD_SPIKE'])
    conds.append(df['hist_corr'] <= config['THRESHOLD_HIST'])
    # Stack into array
    flags = np.vstack(conds).astype(int)
    # Sum across filters
    sum_flags = flags.sum(axis=0)
    # Apply consensus
    pred_drop = (sum_flags >= config['CONSENSUS_K']).astype(int)
    # Ground truth
    gt = df['gt_drop'].values
    tp = np.sum((gt == 1) & (pred_drop == 1))
    fp = np.sum((gt == 0) & (pred_drop == 1))
    fn = np.sum((gt == 1) & (pred_drop == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def main():
    df = load_data()
    best_f1 = 0.0
    best_config = None

    # Generate all combinations
    keys, values = zip(*SEARCH_SPACE.items())
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        f1 = compute_f1_for_config(df, config)
        if f1 > best_f1:
            best_f1 = f1
            best_config = config.copy()

    print("\n=== Optimized Configuration ===")
    for k, v in best_config.items():
        print(f"{k}: {v}")
    print(f"\nAchieved F1-score: {best_f1:.4f}")


if __name__ == "__main__":
    main()

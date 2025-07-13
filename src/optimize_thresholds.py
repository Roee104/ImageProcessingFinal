# src/optimize_thresholds.py

import pandas as pd
import numpy as np
from pathlib import Path

# Evaluation metrics function
def compute_metrics(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

# Load per‐frame metrics and ground truth labels
metrics_csv = Path("logs") / "frame_by_frame.csv"
gt_csv = Path("logs") / "gt_labels.csv"  # Your manual labels file

metrics_df = pd.read_csv(metrics_csv)
gt_df = pd.read_csv(gt_csv)

# Merge on video and frame_idx
data = metrics_df.merge(gt_df, on=["video", "frame_idx"])

# Dictionary to store best thresholds
best_thresholds = {}

# 1. Optimize BLUR threshold
blur_vals = data["var_lap"]
thr_candidates = np.linspace(blur_vals.min(), blur_vals.quantile(0.5), 50)
best_f1 = -1; best_thr = None
for thr in thr_candidates:
    y_pred = (data["var_lap"] <= thr).astype(int)
    _, _, f1 = compute_metrics(data["gt_drop"], y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
best_thresholds["THRESHOLD_BLUR"] = best_thr

# 2. Optimize NOISE threshold
noise_vals = data["noise_metric"]
thr_candidates = np.linspace(noise_vals.quantile(0.5), noise_vals.max(), 50)
best_f1 = -1; best_thr = None
for thr in thr_candidates:
    y_pred = (data["noise_metric"] >= thr).astype(int)
    _, _, f1 = compute_metrics(data["gt_drop"], y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
best_thresholds["THRESHOLD_NOISE"] = best_thr

# 3. Optimize DIFF threshold
diff_vals = data["mean_diff"]  # ensure your df has this column (compute when logging)
thr_candidates = np.linspace(0, diff_vals.quantile(0.9), 50)
best_f1 = -1; best_thr = None
for thr in thr_candidates:
    y_pred = (data["mean_diff"] <= thr).astype(int)
    _, _, f1 = compute_metrics(data["gt_drop"], y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
best_thresholds["THRESHOLD_DIFF"] = best_thr

# 4. Optimize SPIKE threshold
dmean_vals = data["delta_mean"]  # ensure `delta_mean` column was logged (compute mean difference)
thr_candidates = np.linspace(dmean_vals.min(), dmean_vals.max(), 50)
best_f1 = -1; best_thr = None
for thr in thr_candidates:
    y_pred = (data["delta_mean"] >= thr).astype(int)
    _, _, f1 = compute_metrics(data["gt_drop"], y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
best_thresholds["THRESHOLD_SPIKE"] = best_thr

# 5. Optimize HIST threshold
hist_vals = data["hist_corr"].dropna()
thr_candidates = np.linspace(hist_vals.min(), hist_vals.max(), 50)
best_f1 = -1; best_thr = None
for thr in thr_candidates:
    y_pred = (data["hist_corr"] <= thr).fillna(0).astype(int)
    _, _, f1 = compute_metrics(data["gt_drop"], y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
best_thresholds["THRESHOLD_HIST"] = best_thr

# Print recommended thresholds
print("=== Optimized Thresholds ===")
for key, val in best_thresholds.items():
    print(f"{key}: {val:.2f}")



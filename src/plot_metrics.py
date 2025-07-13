# src/plot_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Adjust path if you're running from project_root
log_csv = Path("logs") / "frame_by_frame.csv"
if not log_csv.exists():
    log_csv = Path("../logs") / "frame_by_frame.csv"
if not log_csv.exists():
    raise FileNotFoundError(
        f"Could not find {log_csv}. Run this script from your project root.")

# Read per-frame metrics
df = pd.read_csv(log_csv)

# Convert to numeric and drop NaNs for hist_corr
df['mean_intensity'] = pd.to_numeric(df['mean_intensity'], errors='coerce')
df['var_lap'] = pd.to_numeric(df['var_lap'], errors='coerce')
df['noise_metric'] = pd.to_numeric(df['noise_metric'], errors='coerce')
df['hist_corr'] = pd.to_numeric(df['hist_corr'], errors='coerce')

# Plot histograms for each metric
metrics = [
    ('mean_intensity', 'Mean Intensity'),
    ('var_lap', 'Variance of Laplacian (Blur)'),
    ('noise_metric', 'Noise Metric'),
    ('hist_corr', 'Histogram Correlation')
]

for column, title in metrics:
    plt.figure()
    data = df[column].dropna()
    plt.hist(data, bins=100)
    plt.title(f'Distribution of {title}')
    plt.xlabel(title)
    plt.ylabel('Number of Frames')
    plt.tight_layout()

plt.show()

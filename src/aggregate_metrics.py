"""
src/aggregate_metrics.py

Aggregate per-frame filter metrics into a summary table for each video,
showing total frames, kept frames, and dropped counts per filter category.
Automatically saves the summary to results/summary_table.csv.

Usage:
    cd <project_root>
    python src/aggregate_metrics.py
"""

import pandas as pd
from pathlib import Path


def main():
    # 1. Load the per-frame log
    log_csv = Path("logs") / "frame_by_frame.csv"
    if not log_csv.exists():
        raise FileNotFoundError(f"Log file not found: {log_csv}")

    df = pd.read_csv(log_csv)

    # 2. Count frames by video & filter
    counts = df.groupby(['video', 'filter_dropped'])\
               .size()\
               .unstack(fill_value=0)

    # 3. Ensure 'kept' column
    counts['kept'] = counts.get('kept', 0)

    # 4. Add summary columns
    counts['Total Frames'] = counts.sum(axis=1)
    counts['Kept Frames'] = counts['kept']

    # 5. Rename columns for readability
    counts = counts.rename(columns={
        'corrupted': 'Corrupted',
        'blank_or_white': 'Blank/White',
        'blurry_or_noisy': 'Blurry/Noisy',
        'near_duplicate': 'Near-Duplicate',
        'brightness_spike': 'Brightness-Spike',
        'histogram_outlier': 'Histogram-Outlier'
    })

    # 6. Reorder (and fill missing) columns
    cols = [
        'Total Frames', 'Kept Frames',
        'Corrupted', 'Blank/White', 'Blurry/Noisy',
        'Near-Duplicate', 'Brightness-Spike', 'Histogram-Outlier'
    ]
    for c in cols:
        counts[c] = counts.get(c, 0)
    counts = counts[cols]

    # 7. Print & save
    print("\n=== Frame Drop Summary ===")
    print(counts.to_string())

    out_csv = Path("results") / "summary_table.csv"
    out_csv.parent.mkdir(exist_ok=True)
    counts.to_csv(out_csv, index=True)
    print(f"\nSummary table saved to: {out_csv}")


if __name__ == "__main__":
    main()

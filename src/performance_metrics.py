# src/performance_metrics.py

"""
Compute summary-quality and performance metrics:
1. Frame-reduction ratio per video.
2. File-size reduction raw vs. cleaned.
3. (Optional) Processing speed: requires rerunning main.py with a timer.

Usage:
    cd <project_root>
    python src/performance_metrics.py
"""

import pandas as pd
from pathlib import Path
import os
import time


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main():
    # Paths
    summary_csv = Path("results") / "summary_table.csv"
    raw_dir = Path("data")
    cleaned_dir = Path("results") / "cleaned_videos"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary table not found: {summary_csv}")

    # 1. Frame-reduction ratio
    summary = pd.read_csv(summary_csv, index_col=0)
    summary['Reduction (%)'] = 100 * \
        (1 - summary['Kept Frames'] / summary['Total Frames'])

    # 2. File-size reduction
    sizes = []
    for video in summary.index:
        raw_path = raw_dir / video
        clean_path = cleaned_dir / video
        raw_size = get_file_size_mb(raw_path) if raw_path.exists() else None
        clean_size = get_file_size_mb(
            clean_path) if clean_path.exists() else None
        reduction = None
        if raw_size and clean_size:
            reduction = 100 * (1 - clean_size / raw_size)
        sizes.append((raw_size, clean_size, reduction))
    sizes_df = pd.DataFrame(sizes, index=summary.index, columns=[
                            'Raw Size (MB)', 'Cleaned Size (MB)', 'Size Reduction (%)'])

    # Combine
    perf_df = pd.concat(
        [summary[['Total Frames', 'Kept Frames', 'Reduction (%)']], sizes_df], axis=1)

    # 3. Processing speed (optional)
    print("To measure processing speed, rerun main.py with a timer, e.g.:")
    print("  time python src/main.py")
    print("and record the real elapsed time.\n")

    # Print and save
    print("\n=== Performance Metrics ===")
    print(perf_df.to_string(float_format='%.2f'))

    out_csv = Path("results") / "performance_metrics.csv"
    perf_df.to_csv(out_csv)
    print(f"\nPerformance metrics saved to {out_csv}")


if __name__ == "__main__":
    main()

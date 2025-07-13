# src/visualize_results.py

"""
Phase 5 - Visualization & Illustration

Generates:
1. Stacked bar chart of frames dropped by filter, per video.
2. Timeline plot of blur metric for a representative video with drop events overlaid.
3. Sample-frame gallery: extracts and saves one dropped & one kept frame per filter category.

Usage:
    cd <project_root>
    python src/visualize_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Paths
LOG_CSV = Path("logs") / "frame_by_frame.csv"
RAW_DIR = Path("data")  # Update if raw videos are elsewhere
RESULTS_DIR = Path("results")
FIG_DIR = Path("docs") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def bar_chart(df):
    counts = df.groupby(['video', 'filter_dropped']) \
               .size() \
               .unstack(fill_value=0)
    # Ensure kept exists
    counts['kept'] = counts.get('kept', 0)
    counts = counts.rename(columns={
        'corrupted': 'Corrupted',
        'blank_or_white': 'Blank/White',
        'blurry_or_noisy': 'Blurry/Noisy',
        'near_duplicate': 'Near-Duplicate',
        'brightness_spike': 'Brightness-Spike',
        'histogram_outlier': 'Histogram-Outlier',
        'kept': 'Kept'
    })
    counts.plot(kind='bar', figsize=(12,6))
    plt.title("Frame Counts by Filter and Video")
    plt.ylabel("Frame Count")
    plt.xlabel("Video")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bar_drops.png")
    plt.show()

def timeline_plot(df, video, metric='var_lap'):
    sub = df[df.video == video]
    plt.figure(figsize=(10,4))
    plt.plot(sub.frame_idx, sub[metric], label=metric)
    drops = sub[sub.filter_dropped != 'kept']
    plt.scatter(drops.frame_idx, drops[metric], color='red', label='Dropped', s=10)
    plt.title(f"{metric} over Time for {video}")
    plt.xlabel("Frame Index")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    fname = f"timeline_{video.replace('.mp4','')}.png"
    plt.savefig(FIG_DIR / fname)
    plt.show()

def sample_frames(df, video):
    sample_dir = FIG_DIR / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # list of categories to sample
    cats = df.filter_dropped.unique()
    # For each category, pick first frame and save image
    cap = cv2.VideoCapture(str(RAW_DIR / video))
    for cat in cats:
        subset = df[(df.video == video) & (df.filter_dropped == cat)]
        if subset.empty:
            continue
        idx = int(subset.frame_idx.iloc[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        label = cat.replace(" ", "_")
        out_path = sample_dir / f"{video.replace('.mp4','')}_{label}_{idx}.png"
        cv2.putText(frame, f"{cat} @ {idx}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imwrite(str(out_path), frame)
    cap.release()

def main():
    # Load metrics
    if not LOG_CSV.exists():
        raise FileNotFoundError(f"Missing {LOG_CSV}")
    df = pd.read_csv(LOG_CSV)
    
    # 1. Bar chart
    bar_chart(df)
    
    # 2. Timeline for a representative video
    rep_video = df.video.unique()[0]  # e.g., first in list
    timeline_plot(df, rep_video, metric='var_lap')
    
    # 3. Sample-frame gallery for the same video
    sample_frames(df, rep_video)
    print(f"Saved sample frames in {FIG_DIR / 'samples'}")

if __name__ == "__main__":
    main()


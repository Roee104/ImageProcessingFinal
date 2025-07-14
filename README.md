# Video Summarization via Irrelevant-Frame Filtering

A lightweight, purely classical pipeline for automatic video summarization of street-view and parking-lot footage. By applying a sequence of domain-tuned filters (blank/over-exposed, blur, noise, near-duplicate, brightness-spike, histogram-outlier), this project produces concise yet representative summaries without any deep-learning dependencies.

---

## 📋 Table of Contents

1. [Features](#features)  
2. [Repository Structure](#repository-structure)  
3. [Dependencies](#dependencies)  
4. [Installation](#installation)  
5. [Usage](#usage)  
   1. [1. Preprocessing](#1-preprocessing)  
   2. [2. Initial Cleaning Pass](#2-initial-cleaning-pass)  
   3. [3. (Optional) Calibrate Thresholds](#3-optional-calibrate-thresholds)  
   4. [4. Ground-Truth Labeling](#4-ground-truth-labeling)  
   5. [5. Optimize Thresholds](#5-optimize-thresholds)  
   6. [6. Final Cleaning Pass](#6-final-cleaning-pass)  
   7. [7. Aggregate Metrics](#7-aggregate-metrics)  
   8. [8. Evaluate Performance](#8-evaluate-performance)  
   9. [9. Performance Metrics](#9-performance-metrics)  
   10. [10. Visualization](#10-visualization)  
6. [Results](#results)  
7. [Project Keywords](#project-keywords)  
8. [License](#license)  
9. [Acknowledgments](#acknowledgments)  

---

## 🔑 Features

- **Robust Preprocessing:**  
  Downsamples 60 FPS → 30 FPS, resizes frames to uniform resolution, converts to grayscale, removes audio.

- **Six Sequential Filters:**  
  1. Corrupted/Unreadable  
  2. Blank/Over-Exposed  
  3. Blur (Variance of Laplacian)  
  4. Noise (Median-Difference)  
  5. Frame-Difference (Near-Duplicate)  
  6. Brightness-Spike & Histogram-Outlier

- **Threshold Tuning:**  
  - Percentile-based calibration  
  - F₁-score optimization against manually labeled ground truth

- **Evaluation Suite:**  
  - Ground-truth labeling helper  
  - Confusion matrix & precision/recall/F₁  
  - Frame- and file-size reduction ratios  
  - Processing speed measurement

- **Visualization Toolkit:**  
  - Bar charts of drop counts  
  - Timeline plots with drop-event overlay  
  - Sample-frame gallery per filter category

- **Purely Classical:**  
  No deep-learning dependencies—ideal for real-time or resource-constrained environments.

---

## 📂 Repository Structure

```text
.
├── data/                       
│   ├── abby_road_view.mp4
│   ├── parking_view.mp4
│   └── ...
├── results/
│   ├── cleaned_videos/         
│   ├── summary_table.csv       
│   └── performance_metrics.csv 
├── docs/
│   └── figures/                
├── logs/
│   ├── preprocessing_summary.csv
│   ├── frame_by_frame.csv      
│   └── gt_labels.csv           
├── src/                        
│   ├── preprocess.py
│   ├── filters.py
│   ├── main.py
│   ├── sample_and_label_frames.py
│   ├── optimize_thresholds.py
│   ├── aggregate_metrics.py
│   ├── evaluate.py
│   ├── performance_metrics.py
│   └── visualize_results.py
├── README.md                   
└── LICENSE
````

---

## ⚙️ Dependencies

* **Python 3.7+**
* **OpenCV** (`opencv-python`)
* **NumPy**
* **Pandas**
* **Matplotlib**
* **scikit-learn**

Install via:

```bash
pip install opencv-python numpy pandas matplotlib scikit-learn
```

---

## 🚀 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Roee104/ImageProcessingFinal.git
   cd ImageProcessingFinal
   ```
2. Install dependencies (see above).

---

## 🚀 Usage

Run each step from the **project root**:

### 1. Preprocessing

```bash
python src/preprocess.py
```

* Downsample, resize, grayscale, remove audio
* Outputs `data/processed/` and `logs/preprocessing_summary.csv`

### 2. Initial Cleaning Pass

```bash
python src/main.py
```

* Applies current thresholds
* Outputs cleaned videos in `results/cleaned_videos/` and `logs/frame_by_frame.csv`

### 3. (Optional) Calibrate Thresholds

```bash
python src/calibrate_thresholds.py
```

* Computes percentile recommendations

### 4. Ground-Truth Labeling

```bash
python src/sample_and_label_frames.py --uniform 300 --dropped 200 --kept 200
```

* Interactive labeling to produce `logs/gt_labels.csv`

### 5. Optimize Thresholds

```bash
python src/optimize_thresholds.py
```

* Finds F₁-optimal static thresholds; copy results into `src/filters.py`

### 6. Final Cleaning Pass

```bash
python src/main.py
```

* Regenerates cleaned videos & metrics with optimized thresholds

### 7. Aggregate Metrics

```bash
python src/aggregate_metrics.py
```

* Builds `results/summary_table.csv`

### 8. Evaluate Performance

```bash
python src/evaluate.py
```

* Confusion matrix & precision/recall/F₁ against labels

### 9. Performance Metrics

```bash
python src/performance_metrics.py
```

* Frame-reduction, file-size reduction, speed; outputs `results/performance_metrics.csv`

### 10. Visualization

```bash
python src/visualize_results.py
```

* Generates bar charts, timeline plots, and sample-frame gallery in `docs/figures/`

---

## 📈 Results

See **`results/summary_table.csv`** and **`results/performance_metrics.csv`** for detailed numbers. Visual outputs in `docs/figures/`.

---

## 🔑 Project Keywords

Sampling • Quantization • Frame-Rate Downsampling • Image Resizing • Grayscale Conversion • Global Intensity Thresholding • Laplacian Filter (Variance of Laplacian) • Median Filter (Noise Metric) • Histogram Correlation (Outlier Detection) • Mean Absolute Pixel Difference (Near-Duplicate Removal) • Brightness-Spike Detection • Percentile-Based Threshold Calibration • F₁-Score Optimization • Confusion Matrix • Precision • Recall • F₁-Score • Frame-Reduction Ratio • File-Size Reduction • Processing Speed (FPS) • Python • OpenCV • NumPy • Matplotlib

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Special thanks to **Dr. Cornel Lustig** for his exceptional teaching and guidance in image processing, and to the open-source communities behind **OpenCV**, **NumPy**, and **Matplotlib**. Continuous feedback from classmates and support from friends and family are also gratefully acknowledged.

```


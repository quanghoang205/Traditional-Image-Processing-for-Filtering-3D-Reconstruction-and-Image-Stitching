
# Traditional Computer Vision: Filtering, 3D Reconstruction, and Image Stitching

This repository contains the implementation of classical computer vision techniques for the Midterm Project of the **Computer Vision (INS3155)** course. The project strictly utilizes traditional image processing methods without the use of deep learning or pretrained networks.

## 🚀 Project Overview

The project is divided into three primary components:

1.  **Image Filtering:** Evaluation of spatial filters for noise reduction and enhancement.
2.  **3D Reconstruction:** Recovery of 3D information from uncalibrated stereo image pairs.
3.  **Image Stitching:** Creating a seamless panorama from multiple overlapping views.

-----

## 🛠 Features

### Part A: Image Filtering

  * Implementation of **Mean**, **Gaussian**, and **Median** filters for denoising.
  * Edge enhancement using **Laplacian sharpening**.
  * Quantitative evaluation using **PSNR** (Peak Signal-to-Noise Ratio).

### Part B: 3D Reconstruction

  * Feature matching (SIFT) and **Fundamental Matrix** estimation.
  * Stereo **Rectification** and **Epipolar Lines** visualization.
  * Disparity Map computation using **Semi-Global Block Matching (SGBM)**.
  * 3D Point Cloud projection and interactive visualization using **Plotly**.

### Part C: Image Stitching

  * Keypoint detection and descriptor extraction using **SIFT**.
  * Robust matching with **FLANN** and **RANSAC** outlier rejection.
  * Perspective transformation (Homography warping) and image blending.

-----

## 📂 Project Structure

```text
.
├── part_a_filtering.py      # Script for Image Filtering tasks
├── part_b_reconstruction.py # Script for 3D Reconstruction tasks
├── part_c_stitching.py      # Script for Panorama Stitching tasks
├── 1a.jpg, 2a.jpg           # Stereo image pairs
├── c.jpg, d.jpg, e.jpg, f.jpg # Overlapping images for stitching
└── Midterm_Exam_Report.pdf  # Detailed technical report
```

-----

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed. You can install the necessary dependencies using pip:

```bash
pip install opencv-python numpy matplotlib plotly
```

-----

## 📖 Usage

### Image Filtering

To run the filtering comparison:

```bash
python part_a_filtering.py
```

### 3D Reconstruction

To generate the disparity map and point cloud:

```bash
python part_b_reconstruction.py
```

### Image Stitching

To create the final panorama:

```bash
python part_c_stitching.py
```

-----

## 📊 Results and Analysis

Detailed comparative analysis, including **PSNR** values for filtering and **RANSAC inlier ratios** for stitching, can be found in the [Midterm\_Exam\_Report.pdf](https://www.google.com/search?q=Midterm_Exam_Report.pdf).

## 👤 Author

  * **Hoang Minh Quang** - Student ID: 23070225
  * Vietnam National University - International School (VNU-IS)

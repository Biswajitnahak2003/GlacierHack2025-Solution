# Phase 2: Final Solution - GlacierHack 2025

**Team:** [Wifightforsleep]  
**Status:** Top 15 Shortlist Submission  
**Task:** Multi-class Semantic Segmentation of Glacier Terrain

## 1. Overview
This directory contains the final codebase used to generate the leaderboard submission for Phase 2. The solution utilizes **GlacierNetV2**, a custom Residual U-Net architecture enhanced with Squeeze-and-Excitation (SE) attention blocks to handle the high inter-class similarity between debris-covered ice and background terrain.

## 2. Key Methodology

### A. Model Architecture: GlacierNetV2
* **Backbone:** Custom ResNet-style encoder with 4 downsampling stages.
* **Attention Mechanism:** Integrated **Squeeze-and-Excitation (SE) Blocks** to adaptively recalibrate channel-wise feature responses. This helps the model focus on spectral bands most relevant to ice features.
* **Input:** 5-Channel Satellite Imagery (Bands 1-5).

### B. Training Strategy
* **Loss Function:** A hybrid loss combining **Tversky Loss** (to handle class imbalance) and **Lovasz-Softmax Loss** (to optimize IoU directly).
    * `Loss = 0.4 * Tversky + 0.6 * Lovasz`
* **Optimizer:** AdamW with Cosine Annealing Warm Restarts (`T_0=10`).
* **Augmentation:** Heavy use of `Albumentations` including Grid Distortion, Elastic Transform, and Coarse Dropout to prevent overfitting.
* **Precision:** Trained using Mixed Precision (`GradScaler`) for memory efficiency.

### C. Inference & Post-Processing
* **Normalization:** Robust Min-Max scaling using **2nd and 98th percentiles** to handle lighting outliers in satellite tiles.
* **TTA:** Test Time Augmentation (Horizontal Flip) is averaged during validation.
* **Morphological Cleaning:** Applied `cv2.morphologyEx` (Opening) with a 3x3 kernel to remove noise from the predicted masks.

## 3. Directory Structure
```text
Phase_2_Final/
├── model_training.ipynb           # Complete training pipeline with Augmentation & Loss details
├── solution.py        # Inference script 
├── glaciernet_v2.pth  # Trained Model 
├── loss_plot.png      # Training/Validation MCC Loss Curves
├── prediction_sample.png # Visual Prediction of the 3 Samples
└── README.md          # This file


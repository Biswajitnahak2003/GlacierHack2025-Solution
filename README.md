# GlacierHack 2025 - Top 15 Solution

**Team:** [ WiFightForSleep ]
**Competition:** GlacierHack 2025

This repository contains my solution for the GlacierHack 2025 challenge. 

### 📂 Phase_2_Final (The Main Submission)
This is the code for the **Top 9** solution (Phase 2).
* **Task:** Multi-class Segmentation.
* **Model:** model (ResNet + Squeeze-and-Excitation).
* **Technique:** I used a Tversky+Lovasz Loss combo and added Morphological Post-processing to clean up the masks.
* **Status:** Complete code and weights are included.

### 📂 Phase_1_Ensemble (Archive)
This is the logic I used for the first phase (Binary Segmentation). 
* **Note:** I placed 16th in this phase, so I originally deleted the files to save space. I've restored the code and logic here to show how I trained the ensemble (7 folds x 5 seeds).

## How to use
Check `requirements.txt` for the libraries I used.
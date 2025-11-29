# Phase 1: Binary Segmentation (Archive)

**Status:** Archived

**Task:** Binary Segmentation (Glacier vs. Non-Glacier)

## What happened to the weights?
To be honest, I placed **16th** in Phase 1 (just missing the top 15 cut-off). I assumed I was out of the competition, so I deleted the heavy model weights and training logs to free up storage space.

However, since I made it back in for the finals, I have recreated the training scripts here to show exactly how I approached the first phase.

## My Approach
For Phase 1, I didn't do anything too fancy with the architecture. I relied on "brute force" ensembling to get a high score.

* **Model:** A classic, standard U-Net (written from scratch).
* **Normalization:** I calculated the maximum pixel value for each image and simply divided by that. (Note: I switched to Percentile Clipping in Phase 2 because it works better for outliers).
* **The Strategy:** I used a heavy ensemble.
  * 7-Fold Cross Validation
  * Repeated across 5 different random seeds
  * Total models trained: 35
  * Final prediction was the average of the best models.

## Files in this folder
* `architecture.py`: The custom U-Net code I used.
* `training_logic.py`: The script showing how I set up the 5-seed / 7-fold loop.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
from architecture import ClassicUNet
import os

# --- NOTES ---
# This script recreates the logic I used for Phase 1. 
# Since I deleted the original files after ranking 16th, 
# this serves as a proof of methodology.

# 1. Dataset Logic
class GlacierBinaryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        # Pseudo-code for how I loaded images in Phase 1
        # image = load_tiff(...) 
        
        # --- PHASE 1 NORM ---
        # Unlike Phase 2, I just divided by the max value here.
        # max_val = image.max()
        # if max_val > 0:
        #     image = image / max_val
            
        # return torch.from_numpy(image), torch.from_numpy(mask)
        pass 

    def __len__(self):
        return len(self.image_paths)

# 2. Loss Function
# I found that mixing BCE and Dice worked best for the binary mask.
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        inputs = torch.sigmoid(inputs)
        smooth = 1e-5
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return bce_loss + (1 - dice)

# 3. The Ensemble Strategy
def run_phase_1_training():
    # I ran this heavy loop to squeeze out performance
    SEEDS = [42, 101, 7, 2023, 55] 
    N_FOLDS = 7
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dummy list just to show the loop
    all_images = np.array(["img1", "img2", "img3"]) 
    
    for seed in SEEDS:
        print(f"--- Running Seed: {seed} ---")
        
        # Shuffle with the new seed
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
            print(f"   Training Fold {fold+1}/{N_FOLDS}")
            
            model = ClassicUNet(n_channels=5, n_classes=1).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = DiceBCELoss()
            
            # ... Training loop happened here ...
            
            # I saved the best model for every single fold/seed combo
            # save_name = f"phase1_seed{seed}_fold{fold+1}.pth"

if __name__ == "__main__":
    print("Reconstruction of Phase 1 training logic.")
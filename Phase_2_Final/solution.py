import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import re
import cv2

# --- CONFIG ---
torch.set_num_threads(1) 

# ======================
# 1. MODEL ARCHITECTURE 
# ======================
class SEBlock(nn.Module):
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_ch, in_ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = self.conv(x)
        out = self.se(out) 
        out += self.shortcut(x)
        return F.relu(out)

class GlacierNetV2(nn.Module):
    def __init__(self, n_channels=5, n_classes=4):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = ResBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ResBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = ResBlock(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = ResBlock(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        if x.shape != x4.shape: x = F.interpolate(x, size=x4.shape[2:], mode='bilinear')
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape != x3.shape: x = F.interpolate(x, size=x3.shape[2:], mode='bilinear')
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        if x.shape != x2.shape: x = F.interpolate(x, size=x2.shape[2:], mode='bilinear')
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        if x.shape != x1.shape: x = F.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        return self.outc(x)

# =========================
# 2. SAMPLE CODE HELPERS 
# =========================
def get_tile_id(filename):
    # For files like img001.tif, extract "001"
    match = re.search(r"img(\d+)\.tif", filename)
    if match:
        return match.group(1)  # Returns "001", "002", etc.
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else None

# =================
# 3. MAIN FUNCTION 
# =================
def maskgeration(imagepath, model_path):
    device = torch.device("cpu")
    model = GlacierNetV2(n_channels=5, n_classes=4)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}")
            
    model.to(device)
    model.eval()

    band_tile_map = {band: {} for band in imagepath}
    for band, folder in imagepath.items():
        if not os.path.exists(folder):
            continue
        files = os.listdir(folder)
        for f in files:
            if f.endswith(".tif"):
                tile_id = get_tile_id(f)
                if tile_id:
                    band_tile_map[band][tile_id] = f

    if not imagepath: return {}
    ref_band = sorted(imagepath.keys())[0]
    if ref_band not in band_tile_map or not band_tile_map[ref_band]:
        for b in imagepath.keys():
            if band_tile_map[b]:
                ref_band = b
                break
                
    tile_ids = sorted(band_tile_map[ref_band].keys())
    masks = {}
    
    # Kernel for cleaning
    kernel = np.ones((3, 3), np.uint8)

    # C. Inference Loop
    with torch.no_grad():
        for tile_id in tile_ids:
            # Collect band arrays
            band_arrays = []
            
            for band_name in sorted(imagepath.keys()):
                if tile_id not in band_tile_map[band_name]:
                    continue

                file_path = os.path.join(
                    imagepath[band_name], band_tile_map[band_name][tile_id]
                )

                if not os.path.exists(file_path):
                    continue

                # Load Image
                arr = np.array(Image.open(file_path)) 
                if arr.ndim == 3:
                    arr = arr[..., 0] 
                
                # Convert to float
                arr = arr.astype(np.float32)
                band_arrays.append(arr)

            if len(band_arrays) < 5:
                continue

            # We stack axis=0 for CNN input (Channels, H, W)
            image = np.stack(band_arrays, axis=0) 
            
            # --- Normalization (Percentile Clip) ---
            p02 = np.percentile(image, 2)
            p98 = np.percentile(image, 98)
            image = np.clip(image, p02, p98)
            
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)

            # Predict
            input_tensor = torch.from_numpy(image).float().unsqueeze(0)
            logits = model(input_tensor)
            pred_mask = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
            
            # Clean
            clean_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
            
            # --- MAP OUTPUT VALUES (0, 85, 170, 255) ---
            full_mask = np.zeros_like(clean_mask, dtype=np.uint8)
            full_mask[clean_mask == 1] = 85
            full_mask[clean_mask == 2] = 170
            full_mask[clean_mask == 3] = 255
            
            masks[tile_id] = full_mask

    return masks
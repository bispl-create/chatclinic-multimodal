import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from datetime import datetime
from PIL import Image

# ==========================================
# [1. Helper Functions]
# ==========================================
def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def center_crop_49(img_tensor):
    """Crops the center 49x49 from a given tensor."""
    if img_tensor.ndim == 4:
        _, _, h, w = img_tensor.shape
        start_h, start_w = (h - 49) // 2, (w - 49) // 2
        return img_tensor[:, :, start_h:start_h+49, start_w:start_w+49]
    elif img_tensor.ndim == 2:
        h, w = img_tensor.shape
        start_h, start_w = (h - 49) // 2, (w - 49) // 2
        return img_tensor[start_h:start_h+49, start_w:start_w+49]
    return img_tensor

def load_image_data(path):
    """Loads a .png file as grayscale and pads to 64x64."""
    img = Image.open(path).convert('L')
    data = np.array(img).astype(np.float32)
    
    h, w = data.shape
    ph, pw = 64 - h, 64 - w
    if ph < 0 or pw < 0:
        start_h, start_w = abs(ph)//2, abs(pw)//2
        data = data[start_h:start_h+64, start_w:start_w+64]
    else:
        data = np.pad(data, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)), mode='constant')
    return data

def normalize_01(x):
    """Normalizes the image using 0.1 and 99.9 percentiles."""
    min_v, max_v = np.percentile(x, [0.1, 99.9])
    return np.clip((x - min_v) / (max_v - min_v + 1e-8), 0, 1).astype(np.float32)

# ==========================================
# [2. Dataset Class]
# ==========================================
class InferenceDatasetTwoCond(Dataset):
    def __init__(self, input_dir, cond_dir):
        """
        Loads input images (y) and condition images (c).
        Assumes files have matching filenames in both directories.
        """
        self.files_y = []
        self.files_c = []
        self.filenames = []
        
        file_list = glob.glob(os.path.join(input_dir, '*.png'))
        file_list.sort()
        
        for fpath_y in file_list:
            fname = os.path.basename(fpath_y)
            fpath_c = os.path.join(cond_dir, fname) 
            
            if os.path.exists(fpath_c):
                self.files_y.append(normalize_01(load_image_data(fpath_y)))
                self.files_c.append(normalize_01(load_image_data(fpath_c)))
                self.filenames.append(fname)
            else:
                print(f"[Warning] Condition file missing for {fname} at {fpath_c}. Skipping.")
                
        if len(self.files_y) == 0:
            print("[Warning] No matching input and condition .png files found.")

    def __len__(self): 
        return len(self.files_y)
        
    def __getitem__(self, idx):
        return {
            'y': torch.from_numpy(self.files_y[idx].copy()).float().unsqueeze(0), 
            'c': torch.from_numpy(self.files_c[idx].copy()).float().unsqueeze(0), 
            'name': self.filenames[idx]
        }

# ==========================================
# [3. Model Architecture (FFC-RRDB)]
# ==========================================
class TimestepEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = self.norm(x).view(b, c, -1).permute(0, 2, 1)
        out, _ = self.attn(x_flat, x_flat, x_flat)
        return x + out.permute(0, 2, 1).view(b, c, h, w)

class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth_rate, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x 

class RRDB_Time(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.time_proj = nn.Linear(time_emb_dim, out_c)
        self.db1 = DenseBlock(out_c)
        self.db2 = DenseBlock(out_c)
        self.db3 = DenseBlock(out_c)
    def forward(self, x, t_emb):
        x = self.proj(x)
        t_add = self.time_proj(t_emb)[:, :, None, None]
        x_t = x + t_add 
        out = self.db1(x_t)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x 

class SpectralTransform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels * 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        ffted = torch.fft.rfft2(x, norm="ortho")
        real, imag = ffted.real, ffted.imag
        stacked = torch.cat([real, imag], dim=1) 
        out = self.conv2(self.relu(self.conv1(stacked)))
        real_out, imag_out = torch.chunk(out, 2, dim=1)
        ffted_out = torch.complex(real_out, imag_out)
        output = torch.fft.irfft2(ffted_out, s=(H, W), norm="ortho")
        return output

class FFC_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.conv_local = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1),
            nn.GroupNorm(8, half),
            nn.SiLU(),
            nn.Conv2d(half, half, 3, padding=1)
        )
        self.conv_global = SpectralTransform(channels - half)
        self.mix = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        half = x.shape[1] // 2
        x_local, x_global = x[:, :half], x[:, half:]
        out_l = self.conv_local(x_local)
        out_g = self.conv_global(x_global)
        out = torch.cat([out_l, out_g], dim=1)
        return self.mix(out) + x

class FFC_RRDB_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(base_channels),
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.down1_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1_rrdb = RRDB_Time(base_channels, base_channels, time_dim)
        self.down2_conv = nn.Conv2d(base_channels, base_channels, 4, 2, 1)
        self.down2_rrdb = RRDB_Time(base_channels, base_channels*2, time_dim)
        self.down3_conv = nn.Conv2d(base_channels*2, base_channels*2, 4, 2, 1)
        self.down3_rrdb = RRDB_Time(base_channels*2, base_channels*4, time_dim)
        self.down3_attn = AttentionBlock(base_channels*4) 
        self.bot1_rrdb = RRDB_Time(base_channels*4, base_channels*4, time_dim)
        self.bot_ffc = FFC_Block(base_channels*4)  
        self.bot2_rrdb = RRDB_Time(base_channels*4, base_channels*4, time_dim)
        self.up3_conv = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.up3_rrdb = RRDB_Time(base_channels*4, base_channels*2, time_dim)
        self.up3_attn = AttentionBlock(base_channels*2) 
        self.up2_conv = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.up2_rrdb = RRDB_Time(base_channels*2, base_channels, time_dim)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.down1_conv(x); x1 = self.down1_rrdb(x, t_emb)
        x = self.down2_conv(x1); x2 = self.down2_rrdb(x, t_emb)
        x = self.down3_conv(x2); x = self.down3_rrdb(x, t_emb); x3 = self.down3_attn(x)
        b = self.bot1_rrdb(x3, t_emb); b = self.bot_ffc(b); b = self.bot2_rrdb(b, t_emb)
        x_up = self.up3_conv(b); x_up = torch.cat([x_up, x2], dim=1); x_up = self.up3_rrdb(x_up, t_emb); x_up = self.up3_attn(x_up)
        x_up = self.up2_conv(x_up); x_up = torch.cat([x_up, x1], dim=1); x_up = self.up2_rrdb(x_up, t_emb)
        return self.out_conv(x_up)

# ==========================================
# [4. Main Inference Execution]
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Inference script for the I2SB FFC-RRDB model.")
    parser.add_argument('-i', '--input', type=str, default="./input", 
                        help="Relative path to the Input data directory (.png)")
    parser.add_argument('-c', '--cond', type=str, default="./input_cond", 
                        help="Relative path to the Condition data directory (.png, must match input names)")
    parser.add_argument('-p', '--pth', type=str, default="./best_model.pth", 
                        help="Relative path to the model weights (.pth)")
    default_out_dir = f"../outputs/i2sb/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument('-o', '--output', type=str, default=default_out_dir, 
                        help="Directory to save the inference results")
    parser.add_argument('--seed', type=int, default=None, 
                        help="Optional random seed for reproducible inference")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        print(f"\n>>> Set random seed to {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> Device configuration: {device}")
    
    model = FFC_RRDB_UNet(in_channels=3, base_channels=64).to(device)
    best_model_path = args.pth
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f">>> Successfully loaded weights from '{best_model_path}'")
    else:
        print(f"[Error] Weights not found at '{best_model_path}'. Please check the path.")
        return
        
    model.eval()
    os.makedirs(args.output, exist_ok=True)
    
    test_ds = InferenceDatasetTwoCond(args.input, args.cond)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    if len(test_ds) == 0:
        print("[Error] No data to process. Exiting.")
        return

    results = []
    all_preds = {}
    
    print("\n>>> Start Inference...")
    with torch.no_grad():
        for b in tqdm(test_loader, desc="Processing files"):
            ty, tc = b['y'].to(device), b['c'].to(device)
            fname = b['name'][0]
            
            t_zero = torch.zeros(1, device=device).long()
            t_pred = model(torch.cat([ty, ty, tc], dim=1), t_zero)
            
            t_pred_np = center_crop_49(torch.clamp(t_pred, 0, 1)).squeeze().cpu().numpy().astype(np.float64)

            save_path_npy = os.path.join(args.output, fname.replace('.png', '.npy'))
            np.save(save_path_npy, t_pred_np)

            all_preds[fname] = t_pred_np
            
            results.append((t_pred_np, fname))

    
    # Create a 2-row summary grid image
    # Create a dynamic layout summary grid image for any N inputs
    import math
    import PIL.Image
    
    file_keys = sorted(list(all_preds.keys()))
    num_files = len(file_keys)
    
    if num_files > 0:
        cell_h, cell_w = 49, 49
        pad = 2
        
        # Calculate dynamic grid dimensions (roughly square)
        cols = math.ceil(math.sqrt(num_files))
        rows = math.ceil(num_files / cols)
        
        grid_h = rows * cell_h + pad * (rows + 1)
        grid_w = cols * cell_w + pad * (cols + 1)
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        
        for idx, k in enumerate(file_keys):
            r = idx // cols
            c = idx % cols
            y = pad + r * (cell_h + pad)
            x = pad + c * (cell_w + pad)
            
            img_np = all_preds[k]
            if img_np.shape != (cell_h, cell_w):
                h_c, w_c = img_np.shape
                y_s = max(0, (h_c - cell_h) // 2)
                x_s = max(0, (w_c - cell_w) // 2)
                img_np = img_np[y_s : y_s + cell_h, x_s : x_s + cell_w]
                
            grid[y : y + cell_h, x : x + cell_w] = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        summ_path = os.path.join(args.output, "summary_grid.png")
        PIL.Image.fromarray(grid).save(summ_path)


    print(f"\n>>> Execution Complete! Results saved at: {args.output}")

if __name__ == '__main__':
    main()

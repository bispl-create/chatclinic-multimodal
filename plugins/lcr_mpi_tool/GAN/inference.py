import os
import argparse
import hashlib
import numpy as np
import h5py
import scipy.io as sio
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset

# ==============================================================================
# 0. Set Seed Function
# ==============================================================================
def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==============================================================================
# 1. Texture-Aware Generator Architecture
# ==============================================================================
class SpatialAttention(nn.Module):
    """Attention module focusing on internal texture of the phantom."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)

class ZeroDownResBlock_Attention(nn.Module):
    """Residual block combining dilation and spatial attention."""
    def __init__(self, channels=128, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.in1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.in2 = nn.InstanceNorm2d(channels)
        self.attention = SpatialAttention()

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.drop(out)
        out = self.in2(self.conv2(out))
        out = self.attention(out)
        return identity + out

class GeneratorZeroDownAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=128, num_blocks=8):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        ]
        
        dilations = [1, 2, 4, 8] 
        for i in range(num_blocks):
            d = dilations[i % len(dilations)]
            layers.append(ZeroDownResBlock_Attention(features, dilation=d))
            
        layers.append(nn.Conv2d(features, out_channels, 3, padding=1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ==============================================================================
# 2. Dataset and Utility Functions
# ==============================================================================

def load_mat_data(file_path):
    """Loads a .mat file and handles h5py / loadmat differences."""
    try:
        mat = sio.loadmat(file_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        data = mat[keys[0]]
    except Exception:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            data = f[keys[0]][:]
    return data.astype(np.float32)

def normalize_01(data):
    """Normalizes the array using 0.1% and 99.9% percentiles."""
    p01 = np.percentile(data, 0.1)
    p999 = np.percentile(data, 99.9)
    data_norm = (data - p01) / (p999 - p01 + 1e-8)
    return np.clip(data_norm, 0.0, 1.0)

def adjust_size_64(data):
    """Center crops or zero pads the image to 64x64."""
    h, w = data.shape
    target = 64
    
    if h > target or w > target:
        ch = min(h, target)
        cw = min(w, target)
        start_y = (h - ch) // 2
        start_x = (w - cw) // 2
        data = data[start_y:start_y+ch, start_x:start_x+cw]
        
    h, w = data.shape
    if h < target or w < target:
        pad_y = target - h
        pad_x = target - w
        pad_y_top = pad_y // 2
        pad_y_bot = pad_y - pad_y_top
        pad_x_left = pad_x // 2
        pad_x_right = pad_x - pad_x_left
        data = np.pad(data, ((pad_y_top, pad_y_bot), (pad_x_left, pad_x_right)), mode='constant')
        
    return data

class MPITestDataset(Dataset):
    """Inference Dataset Loader"""
    def __init__(self, test_root):
        self.test_root = test_root
        self.files = sorted([f for f in os.listdir(test_root) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.test_root, file_name)
        
        import PIL.Image
        img_pil = PIL.Image.open(file_path).convert('L')
        img = np.array(img_pil).astype(np.float32) / 255.0
        img = adjust_size_64(img)
        
        img = torch.from_numpy(img.copy()).unsqueeze(0)
        return img, file_name
def center_crop_49(img_array):
    """Center crops the numpy array to 49x49 size."""
    h, w = img_array.shape
    ch, cw = 49, 49
    start_y = (h - ch) // 2
    start_x = (w - cw) // 2
    return img_array[start_y:start_y+ch, start_x:start_x+cw]

def get_file_hash(file_path):
    """Yields MD5 hash of the given file path."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# ==============================================================================
# 3. Inference Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Inference script for the GAN-based MPI image restoration model.")
    parser.add_argument('-i', '--input', type=str, default="./input", 
                        help="Relative path to the Input data directory containing .mat files")
    parser.add_argument('-p', '--pth', type=str, default="./best_model.pth", 
                        help="Relative path to the best_model weights (.pth)")
    # Generate dynamic timestamped output path
    default_out_dir = f"../outputs/gan/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument('-o', '--output', type=str, default=default_out_dir, 
                        help="Directory to save the resulting .npy files")
    parser.add_argument('--seed', type=int, default=None, 
                        help="Optional random seed for reproducible inference")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        print(f"\n>>> Set random seed to {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> Running GAN Inference on Device: {device}")
    
    # 1. Weights Validation
    ckpt_path = args.pth
    if not os.path.exists(ckpt_path):
        print(f"Weight file not found at: {ckpt_path}")
        return
    
    print(f"Loading model weights from: {ckpt_path}")
    print(f"Weight MD5 Hash: {get_file_hash(ckpt_path)}")

    # 2. Load Model 
    model = GeneratorZeroDownAttention().to(device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval()
    
    weight_sum = sum(p.sum().item() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized. Total params: {num_params:,} (Sum: {weight_sum:.6f})\n")

    # 3. Load Dataset using MPITestDataset
    try:
        dataset = MPITestDataset(args.input)
        print(f"Found {len(dataset)} files in '{args.input}'")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if len(dataset) == 0:
        print(f"Warning: Dataset at {args.input} appears empty. No inference will be performed.")
        return

    os.makedirs(args.output, exist_ok=True)
    
    # 4. Inference Loop
    count = 0
    all_preds = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            img_l, file_name = dataset[i]
            x_t = img_l.unsqueeze(0).to(device)
            
            # Forward pass
            out_tensor = model(x_t)
            
            # Post-process
            out_img = out_tensor.squeeze().cpu().numpy()
            out_img = np.clip(out_img, 0.0, 1.0)
            
            # Restore to native 49x49 size
            final_img = center_crop_49(out_img)
            
            # Save numpy array
            save_name = file_name.replace('.png', '.npy')
            save_path = os.path.join(args.output, save_name)
            np.save(save_path, final_img.astype(np.float64))
            all_preds[file_name] = final_img
            
            count += 1
            if count % 10 == 0:
                print(f"Processing... {count}/{len(dataset)}", end='\r')
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

    
    print(f"\nInference fully complete! Results saved in '{args.output}'")

if __name__ == '__main__':
    main()

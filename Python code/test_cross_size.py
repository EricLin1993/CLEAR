"""
test the CLEAR model on cross-size reconstruction
training size: 64x64
test size: 36x36, 48x48, 64x64, 128x128, 256x256, 384x384, 512x512
"""

import torch
import numpy as np
import scipy.io as scio
import os
import time
from pathlib import Path

from models.model import CLEAR

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# === RLNE metric ===
def compute_rlne(x_true, x_pred):
    """Relative L2 Norm Error (real-part relative L2 error on complex spectra)."""
    x_pred = np.asarray(x_pred)
    x_true = np.asarray(x_true)
    l2_error = np.linalg.norm(x_pred - x_true, ord=2)
    l2_true = np.linalg.norm(x_true, ord=2)
    return l2_error / l2_true if l2_true > 0 else 0.0

# === Cross-size inference utilities ===
TILE_SIZE = 64
CHUNK_SIZE = 16

def get_pad_size(h, w, tile_size=TILE_SIZE):
    if h <= tile_size and w <= tile_size:
        return tile_size, tile_size
    import math
    return math.ceil(h / tile_size) * tile_size, math.ceil(w / tile_size) * tile_size

def pad_array(arr, h_orig, w_orig, h_pad, w_pad):
    out = np.zeros((h_pad, w_pad, arr.shape[-1]), dtype=np.float32)
    out[:h_orig, :w_orig, :] = arr[:h_orig, :w_orig, :]
    return out

def split_tiles(tensor, tile_size=TILE_SIZE):
    b, h, w, c = tensor.shape
    nh, nw = h // tile_size, w // tile_size
    x = tensor.view(b, nh, tile_size, nw, tile_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b * nh * nw, tile_size, tile_size, c), nh, nw

def merge_tiles(tiles, batch_size, nh, nw, tile_size=TILE_SIZE):
    c = tiles.shape[-1]
    x = tiles.view(batch_size, nh, nw, tile_size, tile_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, nh * tile_size, nw * tile_size, c)

def forward_conformer_blocks(model, blocks, tiles, chunk_size=CHUNK_SIZE):
    outputs = []
    for i in range(0, tiles.shape[0], chunk_size):
        chunk = tiles[i:i + chunk_size]
        x = model.input_projection(chunk)
        for block in blocks:
            x = block(x)
        outputs.append(model.output_projection(x))
    return torch.cat(outputs, dim=0)

@torch.no_grad()
def cross_size_infer(model, inp, sampled, mask, tile_size=TILE_SIZE, chunk_size=CHUNK_SIZE):
    """<=64: zero-pad in frequency domain; >64: 64x64 tiled Conformer + full-image DC."""
    _, h, w, _ = inp.shape
    _, h_orig, w_orig, _ = sampled.shape
    if h <= tile_size and w <= tile_size:
        out2 = model(inp, sampled, mask)
        return out2[:, :h_orig, :w_orig, :]

    # >64: 64x64 tiled Conformer + full-image DC (exactly as in notebook)
    tiles, nh, nw = split_tiles(inp, tile_size)
    out1_tiles = forward_conformer_blocks(model, model.conformer_block1, tiles, chunk_size)
    out1 = merge_tiles(out1_tiles, 1, nh, nw, tile_size)
    out1 = model.dc_layer(out1, sampled, mask)          # DC after first phase

    tiles, nh, nw = split_tiles(out1, tile_size)        # re-tile after DC correction
    out2_tiles = forward_conformer_blocks(model, model.conformer_block2, tiles, chunk_size)
    out2 = merge_tiles(out2_tiles, 1, nh, nw, tile_size)
    out2 = model.dc_layer(out2, sampled, mask)          # DC after second phase
    return out2[:, :h_orig, :w_orig, :]

# === Main test script ===
if __name__ == '__main__':
    # Hyperparameters (match training config)
    input_dim = 2
    model_dim = 32
    num_heads = 2
    num_layers = 2
    conv_kernel_size = (17, 17)
    dropout = 0.1

    # Load model (use original CLEAR)
    model = CLEAR(
        input_dim=input_dim, model_dim=model_dim, num_heads=num_heads,
        num_layers=num_layers, ff_expansion_factor=4,
        conv_kernel_size=conv_kernel_size, dropout=dropout
    )
    model_path = 'Python code/best_model/best_model.pth'  # update as needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Data directory (user-specified)
    data_root = Path('Python code/Simulation_Data')
    size_dirs = {p.name: p for p in sorted(data_root.iterdir()) if p.is_dir()}

    print('Evaluating sizes:', list(size_dirs.keys()))

    for size_label, directory in size_dirs.items():
        files = sorted([f for f in os.listdir(directory) if f.endswith('.mat')])
        rlne_list = []
        print(f'\nProcessing {size_label}: {len(files)} files\n')
        for fname in files:
            mat = scio.loadmat(os.path.join(directory, fname))
            inp = mat.get('f_2d_nus_2c')
            sampled = mat.get('fid_2d_nus_2c')
            mask = mat.get('mask_2c')
            target = mat.get('f_2d_2c')  # ground-truth fully sampled spectrum
            if inp is None or sampled is None or mask is None or target is None:
                continue
            inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            sampled_t = torch.tensor(sampled, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
            start = time.perf_counter()
            out = cross_size_infer(model, inp_t, sampled_t, mask_t, TILE_SIZE, CHUNK_SIZE)
            elapsed = time.perf_counter() - start
            pred = out.squeeze(0).detach().cpu().numpy()

            # Compute RLNE on complex spectrum
            pred_complex = pred[..., 0] + 1j * pred[..., 1]
            target_complex = target[..., 0] + 1j * target[..., 1]
            rlne = compute_rlne(target_complex, pred_complex)
            rlne_list.append(rlne)
            if int(fname.split('.')[0]) == 1:
                print(f'{size_label}/{fname}  RLNE={rlne:.4f}')
            
        if rlne_list:
            mean_rlne = float(np.mean(rlne_list))
            std_rlne = float(np.std(rlne_list))
            print(f'{size_label} summary: n={len(rlne_list)}, RLNE={mean_rlne:.4f}±{std_rlne:.4f}')
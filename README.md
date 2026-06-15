## Consistency-guided Long-range Enhanced Attention for Recon-struction (CLEAR) of Non-Uniform Sampling NMR


<div align="center">
  <img src="https://github.com/EricLin1993/CLEAR/blob/main/Image.png" width="500">
</div>



CLEAR is a PyTorch implementation for reconstructing nonuniformly sampled (NUS) 3D NMR spectra from hypercomplex NMRPipe data. The model leverages Conformer blocks (MHSA + depthwise separable convolution + FFN) and explicitly interleaves a Data Consistency (DC) layer between phases to preserve fidelity to acquired samples in the time/frequency domains. A complex-valued loss is optimized by computing MSE over the real and imaginary channels.


### Repository structure

```
CLEAR/
├─ Matlab Code/             # MATLAB utilities for simulated dataset generation
├─ NMRPipe Code/            # NMRPipe processing scripts
├─ Python Code/
│  ├─ models/
│  │  ├─ model.py              # Original CLEAR architecture, DC layer, complex MSE
│  │  ├─ model_row_wise_attn.py # Variant using Row-Wise Attention for performance comparison
│  │  └─ model_vit.py           # Variant using ViT-style global self-attention
│  ├─ utils/
│  │  ├─ utils.py              # Training data processing utilities
│  │  ├─ nus_proc_func.py      # NUS data processing and reconstruction helper functions
│  │  └─ NUSCon_metrics.py     # NUSCon evaluation metrics (peak detection, Hausdorff, intensity linearity, TPR/FPR via bipartite matching)
│  ├─ train.py              # Training loop with scheduler and early stopping
│  ├─ test.py               # Script to test trained models and perform reconstruction
│  ├─ test_cross_size.py    # Script for cross-size reconstruction testing (supports >64 inputs) with RLNE evaluation on Simulation_Data
│  ├─ test.ipynb            # Jupyter notebook for reconstruction and optional plotting
│  ├─ best_model/
│  │  └─ best_model.pth     # Pretrained weights for quick demo
│  ├─ ft1_Data/             # Reference fully sampled NMRPipe files (*.ft1)
│  ├─ NUS_Data/             # Example non-uniformly sampled inputs (*.ft1)
│  └─ Recon_Results/        # Demo outputs (*.ft1, *.ft)
```

### Installation

- Python 3.9+; PyTorch with CUDA is recommended (CPU is supported)
- Dependencies: `torch`, `numpy`, `scipy`, `nmrglue`, `matplotlib` (for plots)

PowerShell (Windows):

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install torch numpy scipy nmrglue matplotlib
```

Conda:

```bash
conda create -n CLEAR python=3.10 -y
conda activate CLEAR
pip install torch numpy scipy nmrglue matplotlib
```

### Data preparation

- Fully sampled labels: `Python Code/ft1_data/*_label.ft1`
- Hypercomplex NMRPipe example inputs: `Code/NUS_Data/*_nus.ft1`
- Sampling masks: `Python Code/NUS_Mask/mask_*.mat (key: mask)`
- Training dataset `.mat` files should include the following keys (shape `(H, W, 2)` for `[real, imag]`):
  - `f_2d_nus_2c`: frequency-domain NUS input `(H, W, 2)`
  - `fid_2d_nus_2c`: time-domain sampled signal `(H, W, 2)`
  - `f_2d_2c`: frequency-domain fully sampled target `(H, W, 2)`
  - `mask_2c`: binary sampling mask duplicated along channels `(H, W, 2)`


### Quick start (pretrained checkpoint)

1) Open and run `Python Code/test.ipynb`.
2) The notebook loads the pretrained checkpoint `best_model/best_model.pth`, reconstruct undersampled hypercomplex data for different samples, and save the results as hypercomplex files ,and save the results as hypercomplex files`Python Code/Recon_Results/*_recon.ft1`.
3) Optional: use the provided NMRPipe script to process the indirect dimension of the reconstructed data (`.ft1` → `.ft`), then run the plotting cells to generate contour plots. Example `.ft ` files are included for convenience.

Default working directory is `Python Code/`.

### Training from scratch


**Steps:**
1. Generate simulation data using `Matlab Code/Data_gen_main.m`.
2. Edit `Python code/train.py`:
   - `directory`: folder containing training `.mat` files
   - `model_result_path`: path to save the best model  
   Adjust other hyperparameters as needed (e.g., `model_dim`, `num_heads`, `num_layers`, `conv_kernel_size`, `dropout`, `batch_size`, `learning_rate`, `num_epochs`).
3) Run: 

```bash
cd "Python Code"
python train.py
```

4. After training completes, the best model is saved at `model_result_path`. To test the trained model:
   - Edit `Python code/test.py`:
     - `loaded_model.load_state_dict(torch.load(...))`: point to the model path saved during training
     - `data_type`: select the test data type (e.g., 'ADK08')
     - Ensure directories like `Recon_Results/` exist
   - Run:

```bash
cd "Python Code"
python test.py
```

The test script will load the model, reconstruct NUS data, and save results as NMRPipe format files.

To evaluate reconstruction performance across different input sizes (including sizes larger than the training size of 64×64) and report RLNE metrics:

```bash
cd "Python Code"
python test_cross_size.py
```

Place your simulation `.mat` files (with keys `f_2d_nus_2c`, `fid_2d_nus_2c`, `mask_2c`, `f_2d_2c`) under subfolders of `Python code/Simulation_Data/` named by size (e.g., `32x32/`, `64x64/`, `128x128/`, ...). The script will automatically discover the subdirectories, run tiled inference with proper Data Consistency layers for large inputs, compute RLNE for each sample, and print per-size summary statistics (`mean±std`).

### Evaluation metrics

`Python Code/utils/NUSCon_metrics.py` implements the official NUSCon metrics for quantitative assessment of reconstructed spectra:

- `findpeak_2d(..., MinA=0.02)` – detects reliable peaks on the ground-truth spectrum.
- `nuscon_metrics(...)` – returns four scalars:
  - **Frequency accuracy** (1 − normalized symmetric Hausdorff distance of peak positions)
  - **Intensity linearity** (Pearson correlation of matched peak heights)
  - **True-positive rate** and **false-positive rate** obtained from maximum bipartite matching under a spatial `cutoff`.

These metrics are used by `test_cross_size.py` (and the notebook) to report reconstruction quality across different input sizes.

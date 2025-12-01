## Consistency-guided Long-range Enhanced Attention for Recon-struction (CLEAR) of Non-Uniform Sampling NMR

<br>
<div align="center">
  <img src="https://github.com/EricLin1993/CLEAR/blob/main/Image.png" width="500">
</div>
<br>

### Description
<b>CLEAR</b> is a PyTorch implementation for reconstructing nonuniformly sampled (<b>NUS</b>) 3D NMR spectra from hypercomplex NMRPipe data. The model leverages Conformer blocks (<b>MHSA</b>+ depthwise separable convolution + <b>FFN</b>) and explicitly interleaves a Data Consistency (<b>DC</b>) layer between phases to preserve fidelity to acquired samples in the time/frequency domains. A complex-valued loss is optimized by computing MSE over the real and imaginary channels.


### Repository structure 

```
CLEAR/
├─ Matlab Code/             # MATLAB utilities for simulated dataset generation
├─ NMRPipe Code/            # NMRPipe processing scripts
├─ Python Code/
│  ├─ model.py              # CLEAR architecture, DC layer, complex MSE
│  ├─ utils.py              # Training data process
│  ├─ train.py              # Training loop with scheduler and early stopping
│  ├─ test.ipynb            # Reconstruction and optional plotting
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











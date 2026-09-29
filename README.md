# DDecomp

Core code for **DDecomp: Structured Uncertainty Decomposition Using Masked Autoencoders for Anomaly Detection in Diffusion MRI** (MICCAI 2026).

DDecomp trains a 3D masked autoencoder on healthy diffusion tensor derived mean diffusivity (MD) maps. Stage I learns the reconstruction mean with masked MSE. Stage II initializes from Stage I and jointly fine tunes the mean and log variance output heads with masked Gaussian negative log likelihood.

## Contents

```text
ddecomp/
  mae_mse.py       3D MAE for Stage I
  mae_nll.py       Mean and log variance MAE for Stage II
  data.py          MD volume dataset and preprocessing entry point
  conform.py       Image conformation functions used by the dataset
  inference.py     Joint mask and dropout sampling, variance decomposition
train.py           Training entry point for both stages
test.py            Single-volume testing entry point
requirements.txt  Python dependencies
THIRD_PARTY_NOTICES.md  Per-file origin and license information
third_party_licenses/Apache-2.0.txt  License for conform.py
```

## Data

Each subject has its own directory under `DATA_ROOT`. The loader reads `<subject_id>-dti-Trace-reg-NormMasked.nii.gz` (the MD input filename used in the training data) or `md.nii.gz`. It applies image conformation and linear resizing to `192 × 256 × 256`, then loads the volumes into memory. Prepare the MD maps and registration before training.

## Installation

Use Python 3.8 or newer and install dependencies with `pip install -r requirements.txt` in a suitable PyTorch environment.

## Training

Run from the repository root. The examples use the paper's 3D ViT Large model, batch size 8, and mask ratio 0.75. Paths are supplied at the command line.

```bash
python train.py --stage mse \
  --data-root /path/to/healthy_md \
  --output-dir /path/to/stage1_checkpoints \
  --model large --epochs 4000 --batch-size 8 --amp

python train.py --stage nll \
  --data-root /path/to/healthy_md \
  --pretrained-checkpoint /path/to/stage1_checkpoints/mse_epoch_4000.pth \
  --output-dir /path/to/stage2_checkpoints \
  --model large --epochs 200 --batch-size 8 \
  --mean-learning-rate 1e-5 --logvar-learning-rate 1e-5 --amp
```

## Testing one MD volume

```bash
python test.py \
  --input-md /path/to/subject/md.nii.gz \
  --checkpoint /path/to/stage2_checkpoints/nll_epoch_200.pth \
  --output-dir /path/to/output
```

The script writes predictive mean, aleatoric, epistemic, total, and coverage NIfTI files in the model grid.

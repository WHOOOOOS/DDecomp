from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom

from . import mae_nll
from .conform import check_affine_in_nifti, conform, is_conform

MODEL_BUILDERS = {
    "small": mae_nll.mae_vit_small_patch16_3d,
    "base": mae_nll.mae_vit_base_patch16_3d,
    "large": mae_nll.mae_vit_large_patch16_3d,
}

def load_md(path, image_size):
    source = nib.load(str(path))
    if len(source.shape) != 3:
        raise ValueError("MD input must be a 3D NIfTI volume")
    if not is_conform(source):
        if not check_affine_in_nifti(source):
            raise ValueError(f"Inconsistent NIfTI affine: {path}")
        source = conform(source, order=1, imagetype="image")
    volume = np.asarray(source.get_fdata(), dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"MD input contains non-finite values: {path}")
    factors = np.asarray(image_size, dtype=np.float64) / volume.shape
    if volume.shape != tuple(image_size):
        volume = zoom(volume, factors, order=1)
    if volume.shape != tuple(image_size):
        raise RuntimeError(f"Resized volume shape {volume.shape} differs from {image_size}")
    affine = source.affine.copy()
    affine[:3, :3] /= factors[np.newaxis, :]
    return volume, affine

def load_brain_mask(path, image_size, affine):
    mask = nib.load(str(path))
    if len(mask.shape) != 3:
        raise ValueError("Brain mask must be a 3D NIfTI volume")
    if not check_affine_in_nifti(mask):
        raise ValueError(f"Inconsistent brain-mask affine: {path}")
    aligned = resample_from_to(mask, (tuple(image_size), affine), order=0)
    brain_mask = np.asarray(aligned.get_fdata() > 0, dtype=bool)
    if not brain_mask.any():
        raise ValueError("Brain mask is empty after resampling")
    return brain_mask

def patches_to_volume(patches, image_size, patch_size=16):
    grid = tuple(dim // patch_size for dim in image_size)
    expected = (int(np.prod(grid)), patch_size ** 3)
    if patches.shape != expected:
        raise ValueError(f"Expected patch array {expected}, received {patches.shape}")
    return patches.reshape(*grid, patch_size, patch_size, patch_size).transpose(
        0, 3, 1, 4, 2, 5
    ).reshape(image_size)

def load_model(checkpoint_path, model_name, image_size, dropout_rate, device):
    model = MODEL_BUILDERS[model_name](
        img_size=tuple(image_size), in_chans=1, norm_pix_loss=False,
        dropout_rate=dropout_rate,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.enable_mc_dropout()
    return model

def decompose_uncertainty(model, volume, masks=100, dropout_samples=10,
                          mask_ratio=0.75, seed=42):
    if masks < 1 or dropout_samples < 1 or not 0 < mask_ratio < 1:
        raise ValueError("masks and dropout_samples must be positive; mask_ratio in (0, 1)")
    image_size = tuple(volume.shape)
    patch_size = 16
    if any(size % patch_size for size in image_size):
        raise ValueError("All image dimensions must be multiples of 16")
    n_patches = int(np.prod([size // patch_size for size in image_size]))
    n_mask = n_patches - int(n_patches * (1 - mask_ratio))
    patch_voxels = patch_size ** 3
    sum_mean = np.zeros((n_patches, patch_voxels), dtype=np.float64)
    sum_mean_squared = np.zeros_like(sum_mean)
    sum_variance = np.zeros_like(sum_mean)
    counts = np.zeros(n_patches, dtype=np.int32)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device
    image = torch.from_numpy(np.ascontiguousarray(volume))[None, None].to(device)
    model.eval()
    model.enable_mc_dropout()

    with torch.inference_mode():
        for mask_number in range(masks):
            masked_ids = rng.choice(n_patches, size=n_mask, replace=False)
            patch_mask = torch.zeros((1, n_patches), device=device)
            patch_mask[0, masked_ids] = 1
            for _ in range(dropout_samples):
                latent, used_mask, restore = model.forward_encoder(
                    image, mask_ratio, mask=patch_mask
                )
                mean, logvar = model.forward_decoder(latent, restore)
                if not torch.equal(used_mask, patch_mask):
                    raise RuntimeError("The model returned a different mask")
                mean = mean[0, masked_ids].float().cpu().numpy()
                variance = logvar[0, masked_ids].float().clamp(-6, 2).exp().cpu().numpy()
                sum_mean[masked_ids] += mean
                sum_mean_squared[masked_ids] += mean * mean
                sum_variance[masked_ids] += variance
                counts[masked_ids] += 1
            print(f"mask {mask_number + 1}/{masks}", flush=True)

    if np.any(counts == 0):
        raise RuntimeError("Some patches were never masked; increase --masks")
    divisor = counts[:, None]
    predictive_mean = sum_mean / divisor
    aleatoric = sum_variance / divisor
    epistemic = np.maximum(sum_mean_squared / divisor - predictive_mean ** 2, 0.0)
    total = aleatoric + epistemic
    maps = {
        "predictive_mean": patches_to_volume(predictive_mean.astype(np.float32), image_size),
        "aleatoric": patches_to_volume(aleatoric.astype(np.float32), image_size),
        "epistemic": patches_to_volume(epistemic.astype(np.float32), image_size),
        "total": patches_to_volume(total.astype(np.float32), image_size),
        "coverage": patches_to_volume(
            np.broadcast_to(counts[:, None], sum_mean.shape).astype(np.float32), image_size
        ),
    }
    return maps

def save_maps(maps, affine, output_dir, prefix, brain_mask=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in maps.items():
        if brain_mask is not None and name != "coverage":
            values = np.where(brain_mask, values, 0)
        nib.save(nib.Nifti1Image(values.astype(np.float32), affine),
                 str(output_dir / f"{prefix}_{name}.nii.gz"))

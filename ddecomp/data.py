from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .conform import check_affine_in_nifti, conform, is_conform

def load_and_conform_image(path):
    image = nib.load(str(path))
    if not is_conform(image):
        if not check_affine_in_nifti(image):
            raise ValueError(f"Inconsistent NIfTI affine: {path}")
        image = conform(image, order=1, imagetype="image")
    return np.asarray(image.get_fdata(), dtype=np.float32)

class BrainMDDataset3D(Dataset):

    def __init__(self, data_root, num_subjects=960, target_size=(192, 256, 256)):
        root = Path(data_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {root}")
        self.target_size = tuple(target_size)
        self.samples = []
        subjects = sorted(path for path in root.iterdir() if path.is_dir())[:num_subjects]
        for subject in subjects:
            md_path = subject / f"{subject.name}-dti-Trace-reg-NormMasked.nii.gz"
            if not md_path.is_file():
                md_path = subject / "md.nii.gz"
            if not md_path.is_file():
                raise FileNotFoundError(f"MD volume missing for {subject.name}")
            volume = load_and_conform_image(md_path)
            if volume.shape != self.target_size:
                factors = [new / old for new, old in zip(self.target_size, volume.shape)]
                volume = zoom(volume, factors, order=1)
            self.samples.append((subject.name, volume[np.newaxis].astype(np.float32)))
        if not self.samples:
            raise RuntimeError(f"No subjects found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject_id, volume = self.samples[index]
        return {"subject_id": subject_id, "volume": torch.from_numpy(volume)}

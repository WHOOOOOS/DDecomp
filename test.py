import argparse
from pathlib import Path

import torch

from ddecomp.inference import (
    decompose_uncertainty,
    load_brain_mask,
    load_md,
    load_model,
    save_maps,
)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-md", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Stage-II checkpoint with both output heads")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path)
    parser.add_argument("--output-prefix")
    parser.add_argument("--model", choices=("small", "base", "large"), default="large")
    parser.add_argument("--image-size", type=int, nargs=3, default=(192, 256, 256))
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--masks", type=int, default=100, help="K in the paper")
    parser.add_argument("--dropout-samples", type=int, default=10,
                        help="T per sampled mask in the paper")
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.input_md.is_file():
        parser.error(f"MD input does not exist: {args.input_md}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.brain_mask is not None and not args.brain_mask.is_file():
        parser.error(f"Brain mask does not exist: {args.brain_mask}")
    if any(size < 16 or size % 16 for size in args.image_size):
        parser.error("each image dimension must be a multiple of 16")
    if args.masks < 1 or args.dropout_samples < 1:
        parser.error("--masks and --dropout-samples must be positive")
    if not 0 < args.mask_ratio < 1:
        parser.error("--mask-ratio must be in (0, 1)")
    if not 0 < args.dropout_rate < 1:
        parser.error("--dropout-rate must be in (0, 1) for MC dropout")
    return args

def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    volume, affine = load_md(args.input_md, tuple(args.image_size))
    brain_mask = None
    if args.brain_mask:
        brain_mask = load_brain_mask(args.brain_mask, tuple(args.image_size), affine)
    model = load_model(args.checkpoint, args.model, tuple(args.image_size),
                       args.dropout_rate, torch.device(args.device))
    maps = decompose_uncertainty(model, volume, masks=args.masks,
                                 dropout_samples=args.dropout_samples,
                                 mask_ratio=args.mask_ratio, seed=args.seed)
    input_name = args.input_md.name
    prefix = args.output_prefix or (input_name[:-7] if input_name.endswith(".nii.gz") else args.input_md.stem)
    save_maps(maps, affine, args.output_dir, prefix, brain_mask=brain_mask)
    print(f"Saved maps to {args.output_dir}")

if __name__ == "__main__":
    main()

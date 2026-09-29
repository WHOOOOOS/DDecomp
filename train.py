import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ddecomp.data import BrainMDDataset3D
from ddecomp import mae_mse, mae_nll

MODELS = {
    "small": "mae_vit_small_patch16_3d",
    "base": "mae_vit_base_patch16_3d",
    "large": "mae_vit_large_patch16_3d",
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mse", "nll"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path,
                        help="Stage-I checkpoint required for NLL training")
    parser.add_argument("--model", choices=MODELS, default="large")
    parser.add_argument("--num-subjects", type=int, default=960)
    parser.add_argument("--image-size", type=int, nargs=3, default=(192, 256, 256))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--mean-learning-rate", type=float,
                        help="Stage-II learning rate for the mean output head")
    parser.add_argument("--logvar-learning-rate", type=float,
                        help="Stage-II learning rate for the log-variance output head")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--dropout-rate", type=float,
                        help="Defaults to 0 for MSE and 0.1 for NLL")
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()
    if args.stage == "nll" and args.pretrained_checkpoint is None:
        parser.error("--pretrained-checkpoint is required for --stage nll")
    if args.stage == "nll" and (args.mean_learning_rate is None or args.logvar_learning_rate is None):
        parser.error("Stage II requires --mean-learning-rate and --logvar-learning-rate")
    for name in ("num_subjects", "batch_size", "grad_accumulation", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.mask_ratio < 1:
        parser.error("--mask-ratio must be between 0 and 1")
    if ((args.mean_learning_rate is not None and args.mean_learning_rate <= 0) or
            (args.logvar_learning_rate is not None and args.logvar_learning_rate <= 0)):
        parser.error("output-head learning rates must be positive")
    if any(size < 16 or size % 16 for size in args.image_size):
        parser.error("each --image-size dimension must be divisible by 16")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.dropout_rate is None:
        args.dropout_rate = 0.0 if args.stage == "mse" else 0.1
    if not 0 <= args.dropout_rate < 1:
        parser.error("--dropout-rate must be in [0, 1)")
    if args.epochs is None:
        args.epochs = 4000 if args.stage == "mse" else 200
    if args.learning_rate is None:
        args.learning_rate = 1e-4
    return args

def load_stage_one(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != {"decoder_logvar.weight", "decoder_logvar.bias"}:
        raise RuntimeError(f"Unexpected missing weights: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint weights: {incompatible.unexpected_keys}")

def build_model(args):
    module = mae_mse if args.stage == "mse" else mae_nll
    kwargs = {"img_size": tuple(args.image_size), "in_chans": 1,
              "norm_pix_loss": False, "dropout_rate": args.dropout_rate}
    model = getattr(module, MODELS[args.model])(**kwargs)
    if args.stage == "nll":
        load_stage_one(model, args.pretrained_checkpoint)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for head in (model.decoder_pred, model.decoder_logvar):
            for parameter in head.parameters():
                parameter.requires_grad_(True)
    return model

def make_optimizer(model, args):
    if args.stage == "mse":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, betas=(0.9, 0.95))
    return torch.optim.AdamW([
        {"params": model.decoder_pred.parameters(),
         "lr": args.mean_learning_rate},
        {"params": model.decoder_logvar.parameters(),
         "lr": args.logvar_learning_rate},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95))

def train(args):
    import setproctitle
    setproctitle.setproctitle("zhijia")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    dataset = BrainMDDataset3D(args.data_root, args.num_subjects, tuple(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = build_model(args).to(device)
    optimizer = make_optimizer(model, args)
    amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage={args.stage}; subjects={len(dataset)}; batches={len(loader)}")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            images = batch["volume"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = model(images, mask_ratio=args.mask_ratio)[0]
            scaler.scale(loss / args.grad_accumulation).backward()
            if step % args.grad_accumulation == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
        mean_loss = total_loss / len(loader)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={mean_loss:.6f}", flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            destination = args.output_dir / f"{args.stage}_epoch_{epoch + 1}.pth"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": mean_loss}, destination)

if __name__ == "__main__":
    train(parse_args())

"""Train Ultron's image generator.

Put your training images in one folder per subject:

    data/
      dragon/  dragon1.jpg dragon2.png ...
      cat/     ...

then run:  python train.py --data data
Or try it with no images of your own:  python train.py --cifar10
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ultron.diffusion import Diffusion
from ultron.model import UNet


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="folder with one sub-folder of images per subject")
    src.add_argument("--cifar10", action="store_true", help="download and train on CIFAR-10 (10 subjects)")
    p.add_argument("--out", type=Path, default=Path("checkpoints/ultron.pt"))
    p.add_argument("--image-size", type=int, default=32, choices=[32, 64])
    p.add_argument("--channels", type=int, default=64, help="model width; bigger = better but slower")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}")

    tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # pixels -> [-1, 1]
    ])
    if args.cifar10:
        dataset = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=tf)
    else:
        dataset = datasets.ImageFolder(args.data, transform=tf)
    class_names = dataset.classes
    print(f"{len(dataset)} images, subjects: {', '.join(class_names)}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=len(dataset) > args.batch_size,
                        num_workers=2, pin_memory=device == "cuda")

    config = dict(num_classes=len(class_names), image_size=args.image_size, base_channels=args.channels)
    model = UNet(**config).to(device)
    # EMA weights give noticeably cleaner samples than the raw training weights.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(device=device)
    start_epoch = 0

    if args.resume and args.out.exists():
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            loss = diffusion.loss(model, images, labels, null_label=len(class_names))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for e, m in zip(ema.parameters(), model.parameters()):
                    e.lerp_(m, 1 - 0.999)
            total += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(loader)} loss {loss.item():.4f}")

        print(f"epoch {epoch + 1}/{args.epochs}  avg loss {total / len(loader):.4f}")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "config": config, "class_names": class_names}, args.out)

    print(f"Saved model to {args.out}")
    print(json.dumps({"subjects": class_names}))


if __name__ == "__main__":
    main()

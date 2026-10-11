"""Generate images from a plain-English request.

    python generate.py "Generate me 50 dragon pictures"

Two engines:
  --engine ultron      (default) the diffusion model you trained with train.py.
                       It can only draw subjects it was trained on.
  --engine pretrained  a downloaded pretrained text-to-image model (SD-Turbo).
                       Draws almost anything, no training needed.
"""

import argparse
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from ultron.prompt import match_class, parse_request

MAX_IMAGES = 200


def generate_ultron(subject, count, args, out_dir):
    from ultron.diffusion import Diffusion
    from ultron.model import UNet

    if not args.checkpoint.exists():
        raise SystemExit(f"No trained model at {args.checkpoint}. Run train.py first "
                         f"(or use --engine pretrained).")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    class_names = ckpt["class_names"]
    name = match_class(subject, class_names)
    if name is None:
        raise SystemExit(f"Ultron hasn't learned to draw '{subject}' yet. It knows: {', '.join(class_names)}.\n"
                         f"Add a data/{subject.replace(' ', '_')}/ folder of images and retrain, "
                         f"or use --engine pretrained.")

    model = UNet(**ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = Diffusion(device=args.device)
    label = class_names.index(name)
    size = ckpt["config"]["image_size"]
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    while saved < count:
        n = min(args.batch_size, count - saved)
        labels = torch.full((n,), label, device=args.device, dtype=torch.long)
        images = diffusion.sample(model, labels, size, steps=args.steps, guidance=args.guidance)
        images = (images + 1) / 2
        for img in images:
            saved += 1
            save_image(img, out_dir / f"{saved:04d}.png")
        print(f"  {saved}/{count}")


def generate_pretrained(subject, count, args, out_dir):
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        raise SystemExit("The pretrained engine needs extra packages: pip install diffusers transformers accelerate")

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(args.pretrained_model, torch_dtype=dtype).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = f"a detailed picture of a {subject}, high quality"

    saved = 0
    while saved < count:
        n = min(args.batch_size, count - saved)
        # SD-Turbo is distilled to make a good image in a single step without guidance.
        images = pipe(prompt=[prompt] * n, num_inference_steps=1, guidance_scale=0.0).images
        for img in images:
            saved += 1
            img.save(out_dir / f"{saved:04d}.png")
        print(f"  {saved}/{count}")


def save_grid(out_dir, count):
    from PIL import Image

    paths = sorted(out_dir.glob("[0-9]*.png"))[:count]
    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    cell = min(w, 128)
    cols = min(10, len(images))
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, img in enumerate(images):
        grid.paste(img.resize((cell, cell), Image.LANCZOS), ((i % cols) * cell, (i // cols) * cell))
    grid.save(out_dir / "grid.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request", help='e.g. "Generate me 50 dragon pictures"')
    p.add_argument("--engine", choices=["ultron", "pretrained"], default="ultron")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ultron.pt"))
    p.add_argument("--pretrained-model", default="stabilityai/sd-turbo")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--steps", type=int, default=50, help="denoising steps (ultron engine)")
    p.add_argument("--guidance", type=float, default=3.0, help="how strongly to follow the subject (ultron engine)")
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    req = parse_request(args.request)
    count = min(req.count, MAX_IMAGES)
    if count < req.count:
        print(f"Capping at {MAX_IMAGES} images.")
    out_dir = args.out / f"{req.subject.replace(' ', '_')}_{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"Generating {count} image(s) of '{req.subject}' with the {args.engine} engine on {args.device}...")
    engine = generate_ultron if args.engine == "ultron" else generate_pretrained
    engine(req.subject, count, args, out_dir)
    save_grid(out_dir, count)
    print(f"Done! Images saved in {out_dir}/ (see grid.png for all of them at once)")


if __name__ == "__main__":
    main()

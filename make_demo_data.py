"""Make a tiny synthetic dataset (circles, squares, triangles) to test training quickly.

    python make_demo_data.py && python train.py --data data/shapes --epochs 30
    python generate.py "generate me 50 circle pictures"
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw


def main(out=Path("data/shapes"), per_class=500, size=32):
    for shape in ["circle", "square", "triangle"]:
        folder = out / shape
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            bg = tuple(random.randint(0, 80) for _ in range(3))
            fg = tuple(random.randint(120, 255) for _ in range(3))
            img = Image.new("RGB", (size, size), bg)
            d = ImageDraw.Draw(img)
            r = random.randint(7, 12)
            cx, cy = random.randint(r + 2, size - r - 2), random.randint(r + 2, size - r - 2)
            if shape == "circle":
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
            elif shape == "square":
                d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
            else:
                d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
            img.save(folder / f"{i:04d}.png")
    print(f"Wrote {3 * per_class} images to {out}/")


if __name__ == "__main__":
    main()

# Ultron: an image generator

Type a request in plain English and Ultron makes the pictures:

```bash
python generate.py "Generate me 50 dragon pictures"
```

The images go in `outputs/dragon_<time>/`, along with a `grid.png` that shows all of them at once.

## How it works

Ultron is a **diffusion model**, the same kind of neural network that Stable Diffusion and DALL·E use.

1. **Training:** take a real image, add a random amount of noise to it, and teach the network
   (a U-Net, see `ultron/model.py`) to guess what noise was added. The network is also told which subject the image shows
   ("dragon", "cat", ...), so it learns what each subject looks like.
2. **Generating:** start from pure static and ask the network to remove the noise a little at a time,
   for 50 steps (`ultron/diffusion.py`). A picture of the subject you asked for comes out.
   *Classifier-free guidance* pushes each image to look more like that subject.
3. **The prompt:** `ultron/prompt.py` reads requests like "make fifty pictures of a red dragon plz" and pulls out
   the count (50) and the subject (red dragon).

## Setup

```bash
pip install -r requirements.txt
```

A GPU helps a lot, but everything also runs on a CPU.

## Option A: train your own (the neural net in this repo)

A neural net can only draw what it has seen. To get dragons, give it dragons. Make one folder per subject:

```
data/
  dragon/   dragon001.jpg, dragon002.png, ...   (more is better: aim for 500+)
  castle/   ...
```

Then train it and generate:

```bash
python train.py --data data --epochs 200
python generate.py "Generate me 50 dragon pictures"
```

Other things you can try:

- **A quick test with no images:** `python make_demo_data.py && python train.py --data data/shapes --epochs 30`,
  then `python generate.py "generate me 50 circle pictures"`.
- **A real dataset:** `python train.py --cifar10` downloads 60k photos of 10 subjects
  (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Bigger, sharper images:** `--image-size 64 --channels 128`. This is slower but looks better.
- **Picking up where you left off:** `--resume`.

How long training takes: about 30 epochs on the shapes demo is enough, and that runs fine on a laptop CPU.
CIFAR-10 or your own photos need around 100–300 epochs on a GPU (a free Google Colab GPU works) to look good.

## Option B: pretrained engine (dragons right away, no training)

To draw *anything* without training, use a model someone else has already trained on billions of images:

```bash
pip install diffusers transformers accelerate
python generate.py "Generate me 50 dragon pictures" --engine pretrained
```

This downloads SD-Turbo (about 5 GB) the first time you run it. It takes about 1 second per image on a GPU,
or a few seconds each on a CPU.

## Options for generate.py

| flag | what it does |
|---|---|
| `--engine ultron` / `pretrained` | use your trained model, or the downloaded one |
| `--steps 50` | number of denoising steps (more = slightly better, slower) |
| `--guidance 3.0` | how hard to push toward the subject (try 1–7) |
| `--seed 42` | get the same images again |
| `--batch-size 10` | how many images to make at once (lower it if you run out of memory) |

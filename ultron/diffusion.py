"""Gaussian diffusion: add noise during training, remove it step by step to sample."""

import math

import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, timesteps: int = 1000, device: str | torch.device = "cpu"):
        self.timesteps = timesteps
        self.device = device
        # Cosine schedule (Nichol & Dhariwal, 2021): keeps more signal at early steps.
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alphas_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = (1 - alphas_bar[1:] / alphas_bar[:-1]).clamp(max=0.999)
        self.alphas_bar = torch.cumprod(1 - betas, dim=0).float().to(device)

    def add_noise(self, x0, t, noise):
        a = self.alphas_bar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model, x0, labels, null_label: int, cond_drop: float = 0.1):
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        # Sometimes hide the label so the model also learns unconditional generation.
        drop = torch.rand(labels.shape, device=labels.device) < cond_drop
        labels = torch.where(drop, torch.full_like(labels, null_label), labels)
        return F.mse_loss(model(self.add_noise(x0, t, noise), t, labels), noise)

    @torch.no_grad()
    def sample(self, model, labels, image_size: int, steps: int = 50, guidance: float = 3.0):
        """DDIM sampling with classifier-free guidance. Returns images in [-1, 1]."""
        n = labels.shape[0]
        null = torch.full_like(labels, model.num_classes)
        x = torch.randn(n, 3, image_size, image_size, device=labels.device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=labels.device).long()

        for i, t in enumerate(times):
            t_batch = t.repeat(n)
            eps = model(torch.cat([x, x]), torch.cat([t_batch, t_batch]), torch.cat([labels, null]))
            eps_cond, eps_uncond = eps.chunk(2)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)

            a_t = self.alphas_bar[t]
            a_prev = self.alphas_bar[times[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=x.device)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

        return x

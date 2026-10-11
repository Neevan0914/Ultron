"""A class-conditional U-Net that predicts the noise added to an image."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # The timestep + class embedding scales and shifts every feature map.
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(seq, seq, seq, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 32,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2, 2),
        attention_resolutions: tuple[int, ...] = (16,),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        # Index `num_classes` is the "no class" token used for classifier-free guidance.
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        self.input_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        self.down = nn.ModuleList()
        skip_channels = [base_channels]
        ch, res = base_channels, image_size
        for level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                out_ch = base_channels * mult
                layers = [ResBlock(ch, out_ch, emb_dim, dropout)]
                if res in attention_resolutions:
                    layers.append(SelfAttention(out_ch))
                self.down.append(nn.ModuleList(layers))
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_channels.append(ch)
                res //= 2

        self.mid = nn.ModuleList(
            [ResBlock(ch, ch, emb_dim, dropout), SelfAttention(ch), ResBlock(ch, ch, emb_dim, dropout)]
        )

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            for _ in range(num_res_blocks + 1):
                out_ch = base_channels * mult
                layers = [ResBlock(ch + skip_channels.pop(), out_ch, emb_dim, dropout)]
                if res in attention_resolutions:
                    layers.append(SelfAttention(out_ch))
                self.up.append(nn.ModuleList(layers))
                ch = out_ch
            if level != 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(ch, ch, 3, padding=1)))
                res *= 2

        self.output = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, x, t, labels):
        emb = self.time_mlp(timestep_embedding(t, self.input_conv.out_channels))
        emb = emb + self.class_emb(labels)

        h = self.input_conv(x)
        skips = [h]
        for layer in self.down:
            h = self._run(layer, h, emb)
            skips.append(h)

        h = self._run(self.mid, h, emb)

        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = torch.cat([h, skips.pop()], dim=1)
            h = self._run(layer, h, emb)

        return self.output(h)

    @staticmethod
    def _run(layer, h, emb):
        if not isinstance(layer, nn.ModuleList):
            return layer(h)
        for sub in layer:
            h = sub(h, emb) if isinstance(sub, ResBlock) else sub(h)
        return h

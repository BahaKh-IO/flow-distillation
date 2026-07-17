"""
data.py — the real data we want to imitate.

Each function here IS p_data (the data distribution):
call it, get n fresh samples of shape [n, 2].

The network never sees these formulas — only the points they produce.
"""

import torch


def sample_ring(n: int) -> torch.Tensor:
    """A ring of radius 2 with slight thickness. Returns [n, 2]."""
    theta = torch.rand(n) * 2 * torch.pi          # random angles, uniform
    r = 2.0 + 0.05 * torch.randn(n)               # radius 2 with Gaussian fuzz
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)             # [n, 2], batch first


def sample_moons(n: int) -> torch.Tensor:
    """Two interleaving crescent moons. Returns [n, 2]."""
    half = n // 2

    t_top = torch.rand(half) * torch.pi           # top moon: half circle
    x_top = torch.cos(t_top)
    y_top = torch.sin(t_top)

    t_bot = torch.rand(n - half) * torch.pi       # bottom moon: flipped + shifted
    x_bot = 1 - torch.cos(t_bot)
    y_bot = -torch.sin(t_bot) - 0.3

    x = torch.cat([x_top, x_bot])
    y = torch.cat([y_top, y_bot])
    pts = torch.stack([x, y], dim=1)
    pts = pts + 0.03 * torch.randn_like(pts)      # a little fuzz
    return pts * 1.5


# The shape used by default everywhere. Swap this one line to change dataset.
sample_data = sample_ring


def radius_err(x: torch.Tensor) -> float:
    """Mean |distance from origin - 2.0|, i.e. how far off the true ring.

    Assumes sample_ring's radius (2.0); lower is better, 0 is perfect.
    """
    r = torch.sqrt((x ** 2).sum(dim=1))
    return (r - 2.0).abs().mean().item()

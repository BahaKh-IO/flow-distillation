"""
sampling.py — turning noise into data (inference).

This simulates the ODE with the Euler method (lecture: Algorithm 1):
    X_{t+h} = X_t + h * u_t(X_t),      h = 1/steps

Each loop iteration = ONE DENOISING STEP = one network call.
`steps` here is exactly the thing distillation shrinks (100 -> 1).

Note: t is NOT random here. It marches 0 -> 1 in order, because now we
are actually taking the journey (unlike training, which never does).
"""

import torch

from .model import VelocityNet


@torch.no_grad()  # inference only: no gradients needed anywhere in here
def generate(net: VelocityNet, steps: int, n: int = 2000) -> torch.Tensor:
    """Sample n points from the model using `steps` Euler steps. Returns [n, 2]."""
    x = torch.randn(n, 2)                          # X_0 ~ p_init (the ONLY randomness)
    h = 1.0 / steps                                # step size

    for i in range(steps):
        tval = torch.full((n, 1), i / steps)       # t marches: 0, h, 2h, ..., 1-h
        v = net(x, tval)                           # the arrow u_t(x)
        x = x + v * h                              # one small step along it

    return x                                       # x is now X_1 (hopefully ~ p_data)


@torch.no_grad()
def generate_from(net, x, steps):
    """Same as generate(), but starts from a given x instead of fresh noise."""
    h = 1.0 / steps
    for i in range(steps):
        tval = torch.full((x.shape[0], 1), i / steps)
        x = x + net(x, tval) * h
    return x
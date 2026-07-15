"""
training.py — flow matching training (how the teacher is made).

The whole algorithm (lecture: Algorithm 3, "conditional flow matching"):
  1. z1 ~ p_data        pick a real destination
  2. z0 ~ N(0, I)       pick a random start
  3. t  ~ Uniform[0,1]  pick a random position along the journey
  4. point  = t*z1 + (1-t)*z0     stand partway along the straight line
  5. answer = z1 - z0             the conditional arrow (aim at THIS z1)
  6. regress net(point, t) onto answer with MSE

Key facts to remember:
  - Training is SIMULATION-FREE: no journey is ever walked here.
    One random spot, one arrow, one weight update. That's it.
  - The per-example target is "wrong" (it ignores all other data points),
    but averaging over millions of steps turns it into the correct
    marginal field (Theorem 12). The loss therefore flattens ABOVE zero —
    that floor is the averaging working, not a bug.
"""

import torch

from .model import VelocityNet


def train_flow_matching(
    net: VelocityNet,
    sample_data,                # function n -> [n, 2], e.g. data.sample_ring
    steps: int = 5000,
    batch_size: int = 1000,
    lr: float = 1e-3,
    log_every: int = 500,
) -> VelocityNet:
    """Train `net` with conditional flow matching. Returns the trained net."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for step in range(steps):
        z1 = sample_data(batch_size)              # real destinations   [B, 2]
        z0 = torch.randn(batch_size, 2)           # random starts       [B, 2]
        t = torch.rand(batch_size, 1)             # positions 0..1      [B, 1]

        point = t * z1 + (1 - t) * z0             # a spot on each line [B, 2]
        answer = z1 - z0                          # the correct arrow   [B, 2]

        guess = net(point, t)
        loss = ((guess - answer) ** 2).mean()     # MSE

        optimizer.zero_grad()                     # wipe the notepad
        loss.backward()                           # compute directions (gradients)
        optimizer.step()                          # nudge every weight

        if step % log_every == 0:
            print(f"step {step:5d}   loss {loss.item():.4f}")

    return net

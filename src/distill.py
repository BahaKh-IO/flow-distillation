"""
distill.py — teaching a student to make the ring in ONE leap.

Method: direct trajectory distillation (the simplest form).
  1. z0 ~ N(0, I)              a random start
  2. z_final = teacher's slow, careful 100-step walk from z0
  3. target = z_final - z0     the arrow that lands there in ONE step
  4. regress student(z0, t=0) onto that target

The student's label comes FROM THE TEACHER, not from the data.
That is what makes this distillation.
"""

import torch

from .model import VelocityNet
from .sampling import generate_from


def distill_one_step(
    teacher: VelocityNet,
    student: VelocityNet,
    teacher_steps: int = 100,
    steps: int = 5000,
    batch_size: int = 1000,
    lr: float = 1e-3,
    log_every: int = 500,
) -> VelocityNet:
    """Train `student` to reproduce the teacher's endpoint in a single step."""
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()

    for step in range(steps):
        z0 = torch.randn(batch_size, 2)                      # random starts

        with torch.no_grad():                                # teacher never learns here
            z_final = generate_from(teacher, z0, teacher_steps)

        target = z_final - z0                                # the one-shot arrow

        t0 = torch.zeros(batch_size, 1)                      # student is asked at t=0
        guess = student(z0, t0)
        loss = ((guess - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"step {step:5d}   loss {loss.item():.4f}")

    return student

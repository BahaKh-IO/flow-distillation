"""
Progressive distillation (Salimans & Ho 2022), flow-matching version.

One ROUND: student with N steps learns to match teacher's 2 baby steps in 1.
Then student becomes teacher, N halves, repeat. 64 -> 32 -> ... -> 1.

The paper's Appendix G target formula inverts a DDIM step through the
alpha/sigma schedule. Our sampler is  x = x + v*h, so inverting it is just
    v = (x_end - x_start) / h        # speed = distance / time
"""

import torch

from .model import VelocityNet


def distill_round(teacher, student, sample_data, N, steps=2000,
                  batch_size=1000, lr=1e-3):
    """One round: student (N steps) learns to match teacher (2N steps)."""
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    h = 1.0 / N                                    # student's step size

    for step in range(steps):                      # training steps
        # starting point: real data + noise, at one of the student's N times
        z1 = sample_data(batch_size)
        z0 = torch.randn(batch_size, 2)
        i = torch.randint(0, N, (batch_size, 1))
        t = i.float() * h
        zt = t * z1 + (1 - t) * z0

        # MOVE 3: two baby steps -- only 2 teacher calls, never 100
        with torch.no_grad():
            zt_half = zt + teacher(zt, t) * (h / 2)
            zt_end = zt_half + teacher(zt_half, t + h / 2) * (h / 2)

        # MOVE 4: speed = distance / time
        target = (zt_end - zt) / h

        loss = ((student(zt, t) - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student


def progressive_distill(teacher, sample_data, start_N=64, end_N=1,
                        steps_per_round=2000):
    """Halve repeatedly. Returns (final student, {N: state_dict})."""
    current = teacher
    checkpoints = {}
    N = start_N

    while N >= end_N:                              # rounds
        student = VelocityNet()
        student.load_state_dict(current.state_dict())   # copy the teacher

        print(f"round: {2*N} steps -> {N} steps")
        distill_round(current, student, sample_data, N, steps=steps_per_round)

        checkpoints[N] = {k: v.clone() for k, v in student.state_dict().items()}
        current = student                          # student becomes teacher
        N //= 2                                    # halve

    return current, checkpoints
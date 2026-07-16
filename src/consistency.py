"""
Consistency distillation (Song et al., 2023), flow-matching version.

Idea: every point on a path maps to the SAME endpoint.
Train it by making neighbours agree -- never by computing the endpoint.

  B  = a point on a path
  C  = B + ONE teacher step        <- 1 teacher call
  loss = ( f(B) - f_copy(C) )^2    <- "say the same thing"

Two guardrails, or it collapses:
  - boundary: f is built so f(x, t=1) = x  ->  the endpoint is LOCKED.
    Without it, "everyone says zero" scores perfectly.
  - EMA copy: the target side uses a slow-moving copy, so both sides
    can't drift into agreement on garbage.
"""

import copy

import torch

from .model import VelocityNet


def f(net, x, t):
    """The consistency function: from (x, t), guess where the path ENDS.

    (1-t)*raw + t*x  is the boundary condition:
      t = 1 (data end) -> returns x itself. Network multiplied by zero. LOCKED.
      t = 0 (noise)    -> returns raw. Full freedom.
    """
    raw = net(x, t)
    return (1 - t) * raw + t * x


def consistency_distill(teacher, student, sample_data, steps=30000,
                        batch_size=1000, lr=1e-3, ema_decay=0.999,
                        dt=0.05, log_every=2000):
    """Distil `teacher` into `student` by enforcing self-consistency."""
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()

    # the frozen copy: a SECOND network, never trained, slowly follows student
    student_ema = copy.deepcopy(student)
    for p in student_ema.parameters():
        p.requires_grad = False

    for step in range(steps):
        # --- B: a point on a path, at a RANDOM time (this is why it's flexible)
        z1 = sample_data(batch_size)
        z0 = torch.randn(batch_size, 2)
        t = torch.rand(batch_size, 1) * (1 - dt)      # leave room for one step
        B = t * z1 + (1 - t) * z0

        # --- C: one teacher step later. ONE call. (naive: 100, progressive: 2)
        with torch.no_grad():
            C = B + teacher(B, t) * dt
            t_C = t + dt

        # --- two answers at where the path ends
        answer_B = f(student, B, t)                    # live -- learns
        with torch.no_grad():
            answer_C = f(student_ema, C, t_C)          # frozen copy -- anchor

        # --- "say the same thing"
        loss = ((answer_B - answer_C) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- nudge the copy slowly after the learner (the EMA)
        with torch.no_grad():
            for p_ema, p in zip(student_ema.parameters(), student.parameters()):
                p_ema.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

        if step % log_every == 0:
            print(f"step {step:5d}   loss {loss.item():.5f}")

    return student_ema     # the EMA copy is the better model -- standard practice


@torch.no_grad()
def generate_consistency(net, steps, n=2000):
    """Multistep consistency sampling -- THE DIAL.

    Not a journey. Each call is a COMPLETE answer; between calls you
    partially re-noise and ask again. 1 call = draft, more = refined.
    """
    x = torch.randn(n, 2)
    t0 = torch.zeros(n, 1)
    x = f(net, x, t0)                       # first answer: noise -> ring point

    for i in range(1, steps):               # optional refinement rounds
        tau = 1.0 - i / steps               # how far back to re-noise
        noise = torch.randn(n, 2)
        x = (1 - tau) * x + tau * noise     # knock it partway back to noise
        # after mixing, x's true time label is (1 - tau) -- same convention
        # as everywhere else (t=1 data, t=0 noise; weight-on-noise = 1-t)
        x = f(net, x, torch.full((n, 1), 1 - tau))   # ask again

    return x
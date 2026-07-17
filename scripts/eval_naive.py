"""
Load the naive-distilled student and LOOK at it.

Run from the project root:
    python -m scripts.eval_naive

Requires: checkpoints/teacher.pth, checkpoints/student.pth
Produces: plots/naive_distill_result.png

Includes the teacher itself forced to 1 step (no distillation at all) as a
baseline -- that's the "naive step reduction destroys quality" problem this
whole project exists to fix. And the student pushed past its trained regime
(2, 4 steps) to show it does NOT have a step-count dial, unlike consistency.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import radius_err, sample_data
from src.model import VelocityNet
from src.sampling import generate

TEACHER = "checkpoints/teacher.pth"
STUDENT = "checkpoints/student.pth"


def main():
    teacher = VelocityNet()
    teacher.load_state_dict(torch.load(TEACHER))
    teacher.eval()

    student = VelocityNet()
    student.load_state_dict(torch.load(STUDENT))
    student.eval()

    panels = [
        ("real data", sample_data(2000)),
        ("teacher, 100 steps", generate(teacher, steps=100)),
        ("teacher, 1 step", generate(teacher, steps=1)),
        ("STUDENT, 1 step", generate(student, steps=1)),
        ("student, 2 steps", generate(student, steps=2)),
        ("student, 4 steps", generate(student, steps=4)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    print(f"\n{'panel':>20s} {'|r-2|':>8s}")
    for ax, (title, x) in zip(axes.flat, panels):
        err = radius_err(x)
        ax.scatter(x[:, 0], x[:, 1], s=3)
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"{title}\n|r-2| {err:.3f}", fontsize=10)
        print(f"{title:>20s} {err:8.3f}")

    fig.suptitle("Naive trajectory distillation — did it work?")
    fig.tight_layout()
    fig.savefig("plots/naive_distill_result.png", dpi=150, bbox_inches="tight")
    print("\nplots saved to plots/naive_distill_result.png")


if __name__ == "__main__":
    main()

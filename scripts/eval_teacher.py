"""
Load the saved teacher and LOOK at it.

Run from the project root:
    python -m scripts.eval_teacher

Produces plots in plots/:
  - teacher_100steps.png      the clean ring
  - teacher_step_sweep.png    quality degrading as steps drop (100 -> 1)

Remember the rule learned the hard way: the loss is a liar,
the picture is the judge. This script is the judge.
"""

import matplotlib

matplotlib.use("Agg")  # save figures to files (no window needed)
import matplotlib.pyplot as plt
import torch

from src.model import VelocityNet
from src.sampling import generate

CHECKPOINT = "checkpoints/teacher.pth"


def main():
    teacher = VelocityNet()
    teacher.load_state_dict(torch.load(CHECKPOINT))
    teacher.eval()

    # --- the clean ring at 100 steps ---
    x = generate(teacher, steps=100)
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=3)
    plt.gca().set_aspect("equal")
    plt.title("teacher, 100 steps")
    plt.savefig("plots/teacher_100steps.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- the degradation sweep: the problem statement of the project ---
    sweep = [100, 8, 4, 2, 1]
    fig, axes = plt.subplots(1, len(sweep), figsize=(4 * len(sweep), 4))
    for ax, steps in zip(axes, sweep):
        x = generate(teacher, steps=steps)
        ax.scatter(x[:, 0], x[:, 1], s=3)
        ax.set_aspect("equal")
        ax.set_title(f"{steps} steps")
    fig.suptitle("teacher quality vs number of denoising steps")
    fig.savefig("plots/teacher_step_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("plots saved to plots/")


if __name__ == "__main__":
    main()

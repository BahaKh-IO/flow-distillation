"""
    python -m scripts.consistency_distill
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.consistency import consistency_distill, generate_consistency
from src.data import sample_data
from src.model import VelocityNet

TEACHER = "checkpoints/teacher.pth"
STUDENT = "checkpoints/consistency.pth"


def radius_err(x):
    r = torch.sqrt((x ** 2).sum(dim=1))
    return (r - 2.0).abs().mean().item()


def main():
    teacher = VelocityNet()
    teacher.load_state_dict(torch.load(TEACHER))
    teacher.eval()

    student = VelocityNet()
    student = consistency_distill(teacher, student, sample_data)
    torch.save(student.state_dict(), STUDENT)

    # THE TEST: does quality IMPROVE with more calls? (naive got worse)
    sweep = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, len(sweep), figsize=(3.4 * len(sweep), 3.8))
    print(f"\n{'sampling calls':>16s} {'|r-2|':>8s}")
    for ax, s in zip(axes, sweep):
        x = generate_consistency(student, s)
        ax.scatter(x[:, 0], x[:, 1], s=2)
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"{s} call{'s' if s > 1 else ''}\n|r-2| {radius_err(x):.3f}", fontsize=9)
        print(f"{s:>16d} {radius_err(x):8.3f}")
    fig.suptitle("Consistency model — same weights, quality dial")
    fig.tight_layout()
    fig.savefig("plots/consistency.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
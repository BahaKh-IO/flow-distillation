"""
    python -m scripts.progressive_distill
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import radius_err, sample_data
from src.model import VelocityNet
from src.progressive import progressive_distill
from src.sampling import generate


def main():
    teacher = VelocityNet()
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth"))
    teacher.eval()

    final, ckpts = progressive_distill(teacher, sample_data, start_N=64, end_N=1)
    torch.save(final.state_dict(), "checkpoints/progressive_1step.pth")

    Ns = sorted(ckpts.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.2 * len(Ns), 3.6))
    print(f"\n{'model':>22s} {'|r-2|':>8s}")
    for ax, N in zip(axes, Ns):
        net = VelocityNet(); net.load_state_dict(ckpts[N]); net.eval()
        x = generate(net, N)                       # each student at ITS OWN N
        ax.scatter(x[:, 0], x[:, 1], s=2)
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"{N} steps\n|r-2| {radius_err(x):.3f}", fontsize=9)
        print(f"{f'progressive N={N}':>22s} {radius_err(x):8.3f}")
    fig.tight_layout()
    fig.savefig("plots/progressive.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
"""
Train the teacher on the ring and save its weights.

Run from the project root:
    python -m scripts.train_teacher

Produces: checkpoints/teacher.pth
(The saved .pth file is what "having a teacher" means. No file, no teacher.)
"""

import torch

from src.data import sample_data
from src.model import VelocityNet
from src.training import train_flow_matching

CHECKPOINT = "checkpoints/teacher.pth"


def main():
    teacher = VelocityNet()
    train_flow_matching(teacher, sample_data, steps=5000)

    torch.save(teacher.state_dict(), CHECKPOINT)
    print(f"teacher saved to {CHECKPOINT}")


if __name__ == "__main__":
    main()

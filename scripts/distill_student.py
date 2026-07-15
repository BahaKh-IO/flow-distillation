"""
Distill the trained teacher into a one-step student and save its weights.

Run from the project root:
    python -m scripts.distill_student

Requires: checkpoints/teacher.pth (run train_teacher.py first)
Produces: checkpoints/student.pth
"""

import torch

from src.distill import distill_one_step
from src.model import VelocityNet

TEACHER_CHECKPOINT = "checkpoints/teacher.pth"
STUDENT_CHECKPOINT = "checkpoints/student.pth"


def main():
    teacher = VelocityNet()
    teacher.load_state_dict(torch.load(TEACHER_CHECKPOINT))
    teacher.eval()

    student = VelocityNet()
    distill_one_step(teacher, student)

    torch.save(student.state_dict(), STUDENT_CHECKPOINT)
    print(f"student saved to {STUDENT_CHECKPOINT}")


if __name__ == "__main__":
    main()

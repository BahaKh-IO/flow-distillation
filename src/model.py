"""
model.py — the network. A direction-giver, not a dot-maker.

Input : a point x [B, 2]  +  a time t [B, 1]   (how far along the journey)
Output: a velocity (an arrow) [B, 2]

In lecture notation this network IS u_theta_t(x): the vector field.

IMPORTANT: teacher and student are BOTH instances of this same class.
What makes one a "teacher" and another a "student" is only:
  - which weights it holds (checkpoints/teacher.pth vs checkpoints/student.pth)
  - how it was trained (training.py vs distill.py)
"""

import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()  # switches on nn.Module's bookkeeping (parameter registry)
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),      # 3 = point (2) + time (1)
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),                 # 2 = an arrow; NO activation at the end
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, 2], t: [B, 1] -> glue into [B, 3]
        return self.net(torch.cat([x, t], dim=1))

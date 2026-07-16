# flow-distillation

A toy flow matching model (2D shapes) built to learn generative-model
distillation from the ground up. Everything here is the same skeleton as a
real latent video world model — just at 2 dimensions instead of millions.

## Layout

```
flow-distillation/
├── src/                        the library (importable, reusable code)
│   ├── data.py                 p_data: the shapes we imitate (ring, moons)
│   ├── model.py                VelocityNet — the vector field u_theta(x, t)
│   ├── training.py             flow matching training (how the teacher is made)
│   ├── sampling.py             Euler sampling loop (steps = denoising steps)
│   ├── distill.py              naive distillation: match the teacher's FULL
│   │                           trajectory (e.g. 100 steps) in one leap
│   └── progressive.py          progressive distillation (Salimans & Ho 2022):
│                               repeatedly halve steps, 64 -> 32 -> ... -> 1
├── scripts/                    things you actually run
│   ├── train_teacher.py        trains teacher, saves checkpoints/teacher.pth
│   ├── eval_teacher.py         plots the ring + the step-degradation sweep
│   ├── naive_distill.py        runs distill.py, saves checkpoints/student.pth
│   └── progressive_distill.py  runs progressive.py, saves every halving round
│                               + plots/progressive.png
├── checkpoints/                saved model weights (.pth)
│   ├── teacher.pth             the 100-step teacher
│   ├── student.pth             naive one-leap 1-step student
│   └── progressive_1step.pth   progressively-distilled 1-step student
└── plots/                      saved figures
```

## Why this split (and why there is no teacher.py / student.py)

Teacher and student are the SAME class (`VelocityNet`). What differs is:

- **which weights** each holds  -> different files in `checkpoints/`
- **how each is trained**       -> `training.py` (teacher) vs `distill.py` /
  `progressive.py` (student, two different distillation strategies)

So the code is split by ROLE (data / model / train / sample / distill),
and the teacher/student distinction lives in checkpoints and scripts.

## Naive vs progressive distillation

Both turn the 100-step teacher into a 1-step student, but differently:

- **`distill.py`** — one giant leap. The student is asked to reproduce the
  teacher's *entire* rollout endpoint from a single call. Cheap to write,
  expensive to run (100 teacher calls per training example) and a hard
  regression target, since the student must correct for the whole
  trajectory's curvature at once.
- **`progressive.py`** — repeated halving. Each round only asks a student
  to compress 2 teacher steps into 1 (64->32->...->1), warm-started from
  the previous round's weights. Only 2 teacher calls per training example,
  and each round is a much gentler extrapolation than the naive approach.

## Run

From the project root:

```bash
python -m scripts.train_teacher        # trains + saves the teacher (~30s on CPU)
python -m scripts.eval_teacher         # saves plots: clean ring + step sweep
python -m scripts.naive_distill        # one-leap distillation -> checkpoints/student.pth
python -m scripts.progressive_distill  # halving distillation -> checkpoints/progressive_1step.pth
                                        # + plots/progressive.png (quality at every step count)
```

## Rules learned the hard way

1. The loss is a liar; the picture is the judge. Always plot.
2. A flat loss above zero is the averaging working (Theorem 12), not a bug.
3. Check the zero-output baseline before diagnosing "model collapse".
4. `steps` in sampling = denoising steps = the thing distillation shrinks.

# flow-distillation

A toy flow matching model (2D shapes) built to learn generative-model
distillation from the ground up. Everything here is the same skeleton as a
real latent video world model — just at 2 dimensions instead of millions.

## Layout

```
flow-distillation/
├── src/                  the library (importable, reusable code)
│   ├── data.py           p_data: the shapes we imitate (ring, moons)
│   ├── model.py          VelocityNet — the vector field u_theta(x, t)
│   ├── training.py       flow matching training (how the teacher is made)
│   ├── sampling.py       Euler sampling loop (steps = denoising steps)
│   └── distill.py        distillation methods (TODAY'S WORK — empty)
├── scripts/              things you actually run
│   ├── train_teacher.py  trains teacher, saves checkpoints/teacher.pth
│   └── eval_teacher.py   plots the ring + the step-degradation sweep
├── checkpoints/          saved model weights (.pth) — teacher/student live here
└── plots/                saved figures
```

## Why this split (and why there is no teacher.py / student.py)

Teacher and student are the SAME class (`VelocityNet`). What differs is:

- **which weights** each holds  -> different files in `checkpoints/`
- **how each is trained**       -> `training.py` (teacher) vs `distill.py` (student)

So the code is split by ROLE (data / model / train / sample / distill),
and the teacher/student distinction lives in checkpoints and scripts.

## Run

From the project root:

```bash
python -m scripts.train_teacher   # trains + saves the teacher (~30s on CPU)
python -m scripts.eval_teacher    # saves plots: clean ring + step sweep
```

## Rules learned the hard way

1. The loss is a liar; the picture is the judge. Always plot.
2. A flat loss above zero is the averaging working (Theorem 12), not a bug.
3. Check the zero-output baseline before diagnosing "model collapse".
4. `steps` in sampling = denoising steps = the thing distillation shrinks.

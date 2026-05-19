# Pareto-front analysis

Reconstruct the Pareto front from a multi-objective study and visualize it.

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.samples.sample import SampleStatus
from polarisopt.utils.paths import workspace_layout


def pareto_mask(Y: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    """Boolean mask of non-dominated rows."""
    Y = Y if minimize else -Y
    n = Y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        diff = Y - Y[i]
        dominated = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
        if dominated.any():
            keep[i] = False
    return keep


cfg = load_study_config("study.yaml")
store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
df = store.to_dataframe()
df = df[df["status"] == "finished"]
Y = np.stack([np.asarray(m, dtype=float) for m in df["metric"]])
print(f"{Y.shape[0]} finished samples, {Y.shape[1]} objectives")
```

## 2D Pareto plot

```python
mask = pareto_mask(Y, minimize=True)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(Y[~mask, 0], Y[~mask, 1], c="lightgray", s=20, alpha=0.6, label="dominated")
ax.scatter(Y[mask, 0],  Y[mask, 1],  c="C3", s=40, label=f"Pareto front (n={mask.sum()})")
ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_title("Pareto front")
ax.grid(alpha=0.3)
ax.legend()
```

## 3D Pareto plot

```python
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — needed for projection='3d'

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Y[~mask, 0], Y[~mask, 1], Y[~mask, 2], c="lightgray", s=15, alpha=0.5)
ax.scatter(Y[mask, 0],  Y[mask, 1],  Y[mask, 2],  c="C3", s=40)
ax.set_xlabel("Obj 1"); ax.set_ylabel("Obj 2"); ax.set_zlabel("Obj 3")
```

## Hypervolume over iterations

For 2 objectives, hypervolume is easy to compute exactly:

```python
def hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    pts = points[np.argsort(points[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for p in pts[::-1]:
        if p[1] >= ref[1] or p[0] >= ref[0]:
            continue
        hv += (prev_x - p[0]) * (ref[1] - p[1])
        prev_x = p[0]
    return hv


df = df.sort_values("id").reset_index(drop=True)
ref = np.array([Y[:, 0].max() * 1.1, Y[:, 1].max() * 1.1])
hv = []
for n in range(1, len(df) + 1):
    Y_so_far = Y[:n]
    mask_so_far = pareto_mask(Y_so_far)
    hv.append(hv_2d(Y_so_far[mask_so_far], ref))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(hv) + 1), hv, lw=2)
ax.set_xlabel("evaluation #")
ax.set_ylabel("Pareto hypervolume")
ax.set_title("Hypervolume progression (ref = max·1.1)")
ax.grid(alpha=0.3)
```

For 3+ objectives, use BoTorch's exact hypervolume:

```python
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

ref_t = torch.as_tensor(-ref, dtype=torch.double)
hv_box = Hypervolume(ref_point=ref_t)
hv = []
for n in range(1, len(df) + 1):
    Y_so_far = Y[:n]
    mask_so_far = pareto_mask(Y_so_far)
    front = -torch.as_tensor(Y_so_far[mask_so_far], dtype=torch.double)
    hv.append(float(hv_box.compute(front)))
```

## See also

- [Convergence plots](convergence.md) — single-objective best-so-far
- Concept: [Multi-objective optimization](../concepts/multi-objective.md)

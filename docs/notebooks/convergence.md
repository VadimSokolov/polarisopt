# Convergence plots

Plot the running best-so-far metric over BO iterations to see whether
a sequential study is still making progress.

## Single-objective: best-so-far line

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.samples.sample import SampleStatus
from polarisopt.utils.paths import workspace_layout


def load_finished(study_yaml: str) -> pd.DataFrame:
    cfg = load_study_config(study_yaml)
    store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
    df = store.to_dataframe()
    return df[df["status"] == "finished"].copy()


df = load_finished("study.yaml")
df["objective"] = df["metric"].apply(lambda m: float(m[0]))
df = df.sort_values("id").reset_index(drop=True)
df["best_so_far"] = df["objective"].cummin()

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df.index, df["objective"], s=20, alpha=0.4, label="evaluation")
ax.plot(df.index, df["best_so_far"], color="C1", lw=2, label="best so far")

# Annotate where the warm-up phase ends
warm_end = (df["iteration"] == 0).sum()
ax.axvline(warm_end - 0.5, color="gray", linestyle="--", alpha=0.5)
ax.text(warm_end - 0.5, ax.get_ylim()[1] * 0.95, "  warm-up → BO",
        rotation=0, va="top", color="gray", fontsize=9)

ax.set_xlabel("evaluation #")
ax.set_ylabel("objective")
ax.set_title("BO convergence")
ax.legend()
ax.grid(alpha=0.3)
```

## Batch BO: per-iteration scatter

When ``batch_size > 1``, each BO iteration contributes ``q`` points.
Plot iteration on x-axis with strip-plot semantics:

```python
import seaborn as sns

bo = df[df["iteration"] > 0]
fig, ax = plt.subplots(figsize=(8, 4))
sns.stripplot(data=bo, x="iteration", y="objective", ax=ax, alpha=0.6)
ax.set_title("Per-iteration scatter (batch BO)")
ax.set_ylabel("objective")
ax.grid(alpha=0.3, axis="y")
```

## Wall-clock view

If runtime varies between samples, plot best-so-far against cumulative
wall-clock instead of evaluation count:

```python
df["cum_runtime_h"] = df["runtime_s"].cumsum() / 3600
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["cum_runtime_h"], df["best_so_far"], lw=2)
ax.set_xlabel("cumulative wall-clock (hours)")
ax.set_ylabel("best objective so far")
ax.grid(alpha=0.3)
```

## See also

- [Pareto front](pareto-front.md) — multi-objective convergence
- [Compare two studies](compare-runs.md) — convergence side-by-side

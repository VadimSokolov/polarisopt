# Compare two studies

When you tune hyperparameters or change the metric definition, you'll
want to compare runs side-by-side. polarisopt ships
[`StudyDiff`](../reference/api/studies/diff.md) for the summary, but
notebooks let you go deeper.

## Quick summary (also available as CLI)

```python
from polarisopt.studies.diff import diff_studies

d = diff_studies("baseline.yaml", "tuned.yaml")
print(d.render())
```

```
metric                            baseline         tuned
------------------------------------------------------
samples                                 28            28
finished                                28            27
failed                                   0             1
best metric                          1.234         0.987
```

The same as ``polarisopt diff baseline.yaml tuned.yaml`` at the CLI.

## Convergence overlay

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout


def best_so_far(yaml_path: str) -> pd.DataFrame:
    cfg = load_study_config(yaml_path)
    store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
    df = store.to_dataframe()
    df = df[df["status"] == "finished"].sort_values("id").reset_index(drop=True)
    df["objective"] = df["metric"].apply(lambda m: float(m[0]))
    df["best_so_far"] = df["objective"].cummin()
    df["study"] = cfg.name
    return df


a = best_so_far("baseline.yaml")
b = best_so_far("tuned.yaml")

fig, ax = plt.subplots(figsize=(8, 4.5))
for df, label, color in [(a, a["study"].iloc[0], "C0"),
                         (b, b["study"].iloc[0], "C3")]:
    ax.plot(df.index, df["best_so_far"], lw=2, label=label, color=color)
    ax.scatter(df.index, df["objective"], s=15, alpha=0.3, color=color)
ax.set_xlabel("evaluation #")
ax.set_ylabel("best objective so far")
ax.set_title("Convergence comparison")
ax.legend()
ax.grid(alpha=0.3)
```

## Parameter-space exploration

How differently did the two studies explore the search space? A small
multi-panel scatter:

```python
import itertools

# Combine into one DataFrame with a 'study' column
a["objective"] = a["metric"].apply(lambda m: float(m[0]))
b["objective"] = b["metric"].apply(lambda m: float(m[0]))
combined = pd.concat([a, b], ignore_index=True)
combined[["x1", "x2"]] = combined["inputs"].apply(lambda v: pd.Series(v))

fig, ax = plt.subplots(figsize=(7, 6))
for study_name, sub in combined.groupby("study"):
    ax.scatter(sub["x1"], sub["x2"], s=30, alpha=0.6, label=study_name)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.legend(title="study")
ax.grid(alpha=0.3)
```

## Histogram of final objective

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(a["objective"], bins=20, alpha=0.5, label="baseline", color="C0")
ax.hist(b["objective"], bins=20, alpha=0.5, label="tuned",    color="C3")
ax.set_xlabel("objective")
ax.set_ylabel("count")
ax.set_title("Distribution of evaluated objectives")
ax.legend()
ax.grid(alpha=0.3)
```

## See also

- [`StudyDiff` API](../reference/api/studies/diff.md)
- [Convergence plots](convergence.md)

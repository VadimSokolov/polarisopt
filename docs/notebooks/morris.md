# Morris sensitivity analysis

Rank parameters by their elementary-effects importance after running a
Morris-design static phase. polarisopt produces the (X, Y) data; SALib
computes the analysis.

## Setup

You need a study that ran with a Morris design — see
[the bundled example](../tutorials/02-morris-screening.md) or the
``morris`` example: ``polarisopt examples copy morris ./morris.yaml``.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.analyze import morris as morris_analyze

from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout


cfg = load_study_config("morris.yaml")
store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
df = store.to_dataframe()
df = df[df["status"] == "finished"].sort_values("id").reset_index(drop=True)
print(f"{len(df)} finished samples")
```

## Compute elementary effects

```python
# Get parameter bounds from the YAML (since the store doesn't keep them).
from polarisopt.parameters.space import parameter_space_from_records
space = parameter_space_from_records(cfg.parameters.inline or [])

problem = {
    "num_vars": space.ndim,
    "names": list(space.names),
    "bounds": space.bounds.tolist(),
}
X = np.stack([np.asarray(v) for v in df["inputs"]])
Y = np.array([float(m[0]) for m in df["metric"]])

# num_levels must match what you passed to MorrisDesign(num_levels=...)
Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, num_levels=4)
result = Si.to_df()
print(result.sort_values("mu_star", ascending=False))
```

Output (example):

|     |    mu   | mu_star |  sigma | mu_star_conf |
|-----|--------:|--------:|-------:|-------------:|
| x1  |  3.1    |   8.4   |  10.2  |        2.1   |
| x2  |  4.8    |  12.1   |  14.3  |        3.0   |
| x3  |  0.2    |   1.1   |   1.4  |        0.3   |
| x4  |  0.1    |   0.7   |   0.9  |        0.2   |

``mu_star`` is the canonical Morris importance metric — larger means
more influential.

## Plot: mu_star vs sigma

This is the standard Morris "screening" plot. Variables with large
``mu_star`` and large ``sigma`` are interacting or non-linear; large
``mu_star`` with small ``sigma`` is a clean monotonic effect.

```python
r = result.sort_index()
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(r["mu_star"], r["sigma"], s=80)
for name in r.index:
    ax.annotate(name, (r.loc[name, "mu_star"], r.loc[name, "sigma"]),
                xytext=(5, 5), textcoords="offset points")
ax.set_xlabel(r"$\mu^*$ (mean absolute elementary effect)")
ax.set_ylabel(r"$\sigma$ (std elementary effect)")
ax.set_title("Morris screening")
ax.grid(alpha=0.3)
```

## Use the result to focus BO

Pick the top-K parameters by ``mu_star`` and reduce your search space:

```python
top_k = 2
keep = result.sort_values("mu_star", ascending=False).head(top_k).index.tolist()
print("Calibrate next on:", keep)
```

Then write a follow-up sequential study YAML restricting
``parameters.inline`` to those names.

## See also

- [Tutorial: Morris screening](../tutorials/02-morris-screening.md)
- [`MorrisDesign` API](../reference/api/design/morris.md)

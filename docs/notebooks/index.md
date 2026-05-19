# Notebook gallery

Copy-pasteable analysis recipes built on top of the SampleStore. Each
page is a self-contained workflow you can drop into a Jupyter notebook
or a script.

| Notebook | What it shows |
|---|---|
| [Convergence plots](convergence.md) | Best-so-far over BO iterations, with batch annotations. |
| [Pareto front](pareto-front.md) | Plot non-dominated samples (2D and 3D). |
| [Morris sensitivity](morris.md) | Rank parameters by elementary-effects importance via SALib. |
| [Compare two studies](compare-runs.md) | Side-by-side metrics, parameter exploration, divergence. |

All notebooks assume you've already run at least one study via
``polarisopt run study.yaml`` and have a populated SampleStore at
``<workspace>/polarisopt.db``.

## Common setup

```python
from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.samples.sample import SampleStatus
from polarisopt.utils.paths import workspace_layout
import numpy as np
import pandas as pd

cfg = load_study_config("study.yaml")
store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)

# Pandas DataFrame: one row per sample
df = store.to_dataframe()
df = df[df["status"] == "finished"]
df.head()
```

The DataFrame columns are: ``id``, ``phase``, ``iteration``, ``inputs``,
``status``, ``metric``, ``folder``, ``runtime_s``, ``runner_task_id``,
``message``, ``created_at``, ``updated_at``. Inputs and metrics are
Python lists in DataFrame cells.

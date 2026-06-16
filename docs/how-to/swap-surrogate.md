# How to swap the surrogate

The default surrogate is BoTorch's `SingleTaskGP` (via
[`GPSurrogate`](../reference/api/surrogates/gp.md)). To use a different
model — random forest, MLP, custom kernel — register a new `Surrogate`
subclass.

## What an alternative Surrogate must provide

[`polarisopt.surrogates.base.Surrogate`](../reference/api/surrogates/base.md):

```python
class Surrogate(ABC):
    @property
    @abstractmethod
    def n_objectives(self) -> int: ...

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance), both shaped (n, m)."""
```

That's it. The optimizer (Acquisition) doesn't care how the surrogate
got those numbers as long as they're returned.

## Random-forest surrogate (sklearn)

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from polarisopt.surrogates.base import Surrogate, surrogate_registry

@surrogate_registry.register("rf")
class RandomForestSurrogate(Surrogate):
    def __init__(self, n_estimators: int = 200, *, random_state: int | None = None) -> None:
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        self._n_obj: int | None = None

    @property
    def n_objectives(self) -> int:
        if self._n_obj is None:
            raise RuntimeError("fit() first")
        return self._n_obj

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._n_obj = Y.shape[1]
        # Multi-output regression handled natively by sklearn forests
        self._rf.fit(X, Y if Y.shape[1] > 1 else Y.ravel())

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Mean: bagged ensemble mean. Variance: per-tree spread.
        per_tree = np.stack([t.predict(X) for t in self._rf.estimators_])
        if per_tree.ndim == 2:
            per_tree = per_tree[..., None]   # (n_trees, n, 1)
        mean = per_tree.mean(axis=0)
        var = per_tree.var(axis=0).clip(min=1e-9)
        return mean, var
```

YAML:

```yaml
generator:
  type: acquisition
  options:
    surrogate:
      type: rf
      options:
        n_estimators: 300
        random_state: 42
    acquisition: { type: qei, options: { mc_samples: 128 } }
```

## Caveats with non-GP surrogates

BoTorch's qLogEI / qLogEHVI internals expect a BoTorch `Model` object
with a usable `posterior(X)` method. If you're swapping in a non-GP
surrogate **and** you want to keep using BoTorch acquisitions, wrap your
surrogate in `botorch.models.model.Model` so its `posterior` returns
a `botorch.posteriors.Posterior`. Otherwise, register a custom
Acquisition that does its own optimization against your surrogate's
`predict` directly.

For most v0.x users: stick with `GPSurrogate`. Custom surrogates are a
v1+ scenario.

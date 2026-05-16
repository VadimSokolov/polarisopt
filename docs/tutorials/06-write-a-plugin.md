# 06 · Writing a plugin

Walks through adding a custom **Design** plugin from scratch. The same
pattern works for Surrogates, Acquisitions, Generators, Stop criteria,
Metrics, Simulators, Runners, and Transfers — every pluggable family
has a `Registry` instance and a `make_xxx` factory.

We'll build `RandomGridDesign`: an `n_per_dim`-spaced grid over the
parameter space, jittered by a small noise.

## 1. The ABC

[`polarisopt.design.base.Design`](../reference/api/design/base.md):

```python
class Design(ABC):
    @abstractmethod
    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        ...
```

A Design takes a `ParameterSpace` and an RNG, returns a `(n_points, d)`
array. That's the entire contract.

## 2. Implement

Put this in `my_plugins.py` somewhere on your Python path:

```python
# my_plugins.py
import numpy as np
from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("random_grid")
class RandomGridDesign(Design):
    """Tensor-product grid with small per-axis jitter."""

    def __init__(self, n_per_dim: int, *, jitter: float = 0.02) -> None:
        if n_per_dim < 2:
            raise ValueError(f"n_per_dim must be >= 2, got {n_per_dim}")
        if not 0.0 <= jitter < 0.5:
            raise ValueError(f"jitter must be in [0, 0.5), got {jitter}")
        self.n_per_dim = int(n_per_dim)
        self.jitter = float(jitter)

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        bounds = space.bounds
        axes = [np.linspace(lo, hi, self.n_per_dim) for lo, hi in bounds]
        grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(space.ndim, -1).T
        if self.jitter > 0:
            widths = (bounds[:, 1] - bounds[:, 0]) / (self.n_per_dim - 1)
            grid = grid + rng.uniform(-self.jitter, self.jitter, size=grid.shape) * widths
        return space.clip(grid)
```

## 3. Test it

```python
# test_my_plugins.py
import numpy as np
from polarisopt.design import design_registry
from polarisopt.parameters import Parameter, ParameterSpace

import my_plugins  # noqa: F401  — triggers the @register decorator

def test_random_grid_in_bounds():
    space = ParameterSpace.from_iterable([
        Parameter("x", "a.json", 0.0, 1.0),
        Parameter("y", "a.json", -1.0, 1.0),
    ])
    cls = design_registry.get("random_grid")
    design = cls(n_per_dim=3, jitter=0.05)
    pts = design.generate(space, rng=np.random.default_rng(0))
    assert pts.shape == (9, 2)
    bounds = space.bounds
    assert np.all(pts >= bounds[:, 0])
    assert np.all(pts <= bounds[:, 1])
```

```bash
pytest test_my_plugins.py
```

## 4. Use it from YAML

Reference the plugin by the name you registered it as (`random_grid`):

```yaml
phases:
  - name: grid
    type: static
    design:
      type: random_grid
      options:
        n_per_dim: 5
        jitter: 0.05
```

**Important**: the registry only knows about plugins that have been
imported. For your plugin module to load before the CLI runs, either:

```bash
# Option A: PYTHONPATH + a small wrapper script
PYTHONPATH=. python -c "import my_plugins; from polarisopt.cli import main; main()" \
    run study.yaml
```

```bash
# Option B: import my_plugins at the top of a small driver
python -c "
import my_plugins   # registers
from polarisopt.cli import main
main()
" run study.yaml
```

A future polarisopt version will support entry-points-based plugin
discovery so external packages can just `pip install` and be picked up
automatically.

## 5. Documenting your plugin for the API site

Use NumPy-style docstrings (Parameters / Returns / Raises / Examples).
mkdocstrings will render them under `reference/api/...` if you point
its `paths` at your package. See
[plugin authoring reference](../plugins.md) for the full convention.

## 6. Sharing your plugin

Once it's useful to multiple users, ship it as a separate PyPI package
(``polarisopt-myplugins``) that depends on ``polarisopt``. Drop your
Design / Surrogate / Metric subclasses in the package; users `pip
install polarisopt-myplugins` + `import polarisopt_myplugins` and the
registrations happen at import time.

## See also

- [Plugin authoring reference](../plugins.md)
- [Concept: Plugin registries](../concepts/plugin-registries.md)
- [Registry API](../reference/api/utils/registry.md)

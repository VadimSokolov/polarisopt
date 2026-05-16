# Plugin authoring

Every swappable piece of `polarisopt` is an ABC backed by a plugin
registry. Adding a new algorithm or infrastructure backend is three
steps:

1. Subclass the relevant ABC.
2. Register it via the registry decorator.
3. Reference it from YAML by name.

## Anatomy of a registry

```python
from polarisopt.utils.registry import Registry

class MyABC(ABC):
    @abstractmethod
    def do(self): ...

my_registry: Registry[MyABC] = Registry("my_family")

# YAML factory
def make_my(spec):
    cls = my_registry.get(spec["type"])
    return cls(**(spec.get("options") or {}))
```

Subclasses register on import:

```python
@my_registry.register("foo")
class FooImpl(MyABC):
    def do(self): ...
```

YAML:

```yaml
something:
  type: foo
  options: { ... }
```

## Adding a new Design

Implement
:class:`~polarisopt.design.base.Design` and register it.

```python
import numpy as np
from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace

@design_registry.register("center_grid")
class CenterGridDesign(Design):
    def __init__(self, n_per_dim: int):
        self.n_per_dim = int(n_per_dim)

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator):
        axes = [
            np.linspace(low, high, self.n_per_dim)
            for low, high in space.bounds
        ]
        grid = np.array(np.meshgrid(*axes)).reshape(space.ndim, -1).T
        return space.clip(grid)
```

```yaml
phases:
  - name: grid
    type: static
    design: { type: center_grid, options: { n_per_dim: 5 } }
```

## Adding a new Surrogate

```python
from polarisopt.surrogates.base import Surrogate, surrogate_registry

@surrogate_registry.register("rf")
class RandomForestSurrogate(Surrogate):
    def __init__(self, n_estimators: int = 200):
        from sklearn.ensemble import RandomForestRegressor
        self._rf = RandomForestRegressor(n_estimators=n_estimators)
        self._m = None

    @property
    def n_objectives(self): return self._m

    def fit(self, X, Y):
        self._m = Y.shape[1]
        self._rf.fit(X, Y if self._m > 1 else Y.ravel())

    def predict(self, X):
        import numpy as np
        mean = self._rf.predict(X)
        if mean.ndim == 1: mean = mean[:, None]
        var = np.full_like(mean, 1.0)  # placeholder uncertainty
        return mean, var
```

## Adding a new Runner

```python
from polarisopt.runners.base import Runner, JobSpec, Job, JobStatus, runner_registry

@runner_registry.register("pbs")
class PBSRunner(Runner):
    def submit(self, spec: JobSpec) -> Job: ...
    def status(self, job: Job) -> Job: ...
    def cancel(self, job: Job) -> Job: ...
```

The Slurm runner's [shell-injectable design](https://github.com/anl-polaris/polaris-hpc/blob/master/src/polarisopt/runners/slurm.py)
is a good template: pass a callable to ``__init__`` that takes
``list[str]`` (the argv) and returns a ``subprocess.CompletedProcess``,
so tests can fake the cluster entirely.

## Adding a new Metric

```python
from polarisopt.metrics.base import Metric, metric_registry

@metric_registry.register("vmt")
class VMTMetric(Metric):
    @property
    def n_objectives(self): return 1
    def compute(self, output):
        # output["result_path"] points to a POLARIS HDF5
        ...
```

## Distributing plugins as separate packages

For 0.2+, plugin discovery via Python entry points will be supported so
external packages can ship plugins. For now, install the plugin package
in the same environment as `polarisopt` and import it before invoking
the CLI (e.g. add `import my_polarisopt_plugins` to a sitecustomize, or
wrap the CLI in a thin entry script).

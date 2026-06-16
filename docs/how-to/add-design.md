# How to add a new design

## Steps

1. Subclass [`polarisopt.design.base.Design`](../reference/api/design/base.md).
2. Implement `generate(space, *, rng) -> np.ndarray` returning a
   ``(n_points, ndim)`` matrix already clipped to ``space``.
3. Register with `@design_registry.register("my_name")`.
4. Import the module before invoking the CLI so the registration runs.

## Template

```python
import numpy as np
from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace

@design_registry.register("my_design")
class MyDesign(Design):
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        self.n = int(n)

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        bounds = space.bounds
        unit = rng.uniform(size=(self.n, space.ndim))
        scaled = unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        return space.clip(scaled)
```

## Use from YAML

```yaml
phases:
  - name: my-phase
    type: static
    design:
      type: my_design
      options:
        n: 24
```

## Testing

```python
from polarisopt.design import design_registry
from polarisopt.parameters import Parameter, ParameterSpace
import numpy as np
import my_designs  # noqa  — registers

def test_shape():
    space = ParameterSpace.from_iterable([
        Parameter("x", "a.json", 0.0, 1.0),
    ])
    pts = design_registry.get("my_design")(n=5).generate(
        space, rng=np.random.default_rng(0)
    )
    assert pts.shape == (5, 1)
```

For a full plugin walkthrough see
[Tutorial 06 · Writing a plugin](../tutorials/06-write-a-plugin.md).

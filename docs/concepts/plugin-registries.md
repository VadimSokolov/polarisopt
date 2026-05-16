# Plugin registries

Every swappable piece of polarisopt is an ABC backed by a
[`Registry`](../reference/api/utils/registry.md) instance. The registry
maps short string names (used in YAML) to concrete classes.

## Why a registry, not just `importlib`?

Three reasons.

**YAML-first configuration.** Users describe what they want by name
(``type: lhs``, ``type: qei``). The CLI / pydantic loader doesn't have
to enumerate every concrete class at type-check time — it just hands
the name to the registry.

**Plugins can self-register.** A user writes a class, decorates it with
``@registry.register("my_thing")``, and from then on every YAML and API
call referring to `my_thing` finds it. No editing core code.

**Discoverability.** ``registry.names()`` lists every plugin in a
family. The CLI can print available choices on misconfiguration.

## The contract

```python
from polarisopt.utils.registry import Registry

# 1. Each family owns one Registry instance.
class DesignABC: ...
design_registry: Registry[DesignABC] = Registry("design")

# 2. Concrete classes register at import time.
@design_registry.register("lhs")
class LHSDesign(DesignABC): ...

# 3. Lookups produce the class.
design_registry.get("lhs")  # → LHSDesign

# 4. A make_xxx() factory wraps registry + options.
def make_design(spec: dict):
    cls = design_registry.get(spec["type"])
    return cls(**(spec.get("options") or {}))
```

## All registries in polarisopt

| Family | Registry | Built-ins |
|---|---|---|
| Design | `design_registry` | `lhs`, `morris`, `sobol`, `manual` |
| Simulator | `simulator_registry` | `mock`, `polaris` |
| Metric | `metric_registry` | `identity`, `link_moe`, `choice_share` |
| Runner | `runner_registry` | `local`, `slurm` |
| Transfer | `transfer_registry` | `local`, `anl` |
| Surrogate | `surrogate_registry` | `gp` |
| Acquisition | `acquisition_registry` | `ei`, `qei`, `qehvi` |
| Generator | `generator_registry` | `random`, `acquisition` |
| Stop | `stop_registry` | `max_iter`, `epsilon`, `plateau`, `hypervolume`, `any`, `all` |

## When `@register` runs

The decorator runs at module import time. So for the registry to know
about a plugin, the plugin's module must be imported before the
registry is queried.

For built-ins this happens via the package ``__init__.py`` files —
``polarisopt.design.__init__`` imports `lhs`, `morris`, etc. so just
``import polarisopt.design`` registers everything.

For user plugins, you must import the plugin module before calling the
CLI or any factory:

```python
import my_plugins   # ← triggers @register

from polarisopt.cli import main
main()
```

Future versions will support Python entry-points so external packages
register automatically on `pip install`.

## Why ABCs, not protocols?

Pluggable families are formal ABCs (`abc.ABC` + `@abstractmethod`) so:

- Static type checkers can validate concrete implementations.
- The error message when someone forgets to implement a method is
  immediate (`TypeError: Can't instantiate abstract class X`).
- `isinstance(x, ABC)` works for cases like
  ``if not isinstance(surrogate, GPSurrogate): raise...``.

Protocols would be lighter but lose the `__init_subclass__` hook used
by some registries for self-registration patterns.

## See also

- [Registry API](../reference/api/utils/registry.md)
- [Tutorial 06 · Writing a plugin](../tutorials/06-write-a-plugin.md)
- [Plugin authoring reference](../plugins.md)

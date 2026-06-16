"""Entry-points-based plugin discovery.

External packages can ship plugins for any polarisopt registry by
declaring entry points in their ``pyproject.toml``. polarisopt
auto-discovers them at startup so users only need to ``pip install``
the plugin package — no ``import my_plugins`` boilerplate before
invoking the CLI.

Convention
----------

The entry-point **group name** must be one of:

- ``polarisopt.designs``
- ``polarisopt.surrogates``
- ``polarisopt.acquisitions``
- ``polarisopt.generators``
- ``polarisopt.stops``
- ``polarisopt.metrics``
- ``polarisopt.simulators``
- ``polarisopt.runners``
- ``polarisopt.transfers``

Each entry point's *value* points to a module whose import triggers the
``@<family>_registry.register("name")`` decorators. The entry point's
*name* is informational (we show it on plugin-load failure).

Example external plugin package ``polarisopt-myplugins``:

```toml
[project.entry-points."polarisopt.designs"]
my_grid = "polarisopt_myplugins.designs"

[project.entry-points."polarisopt.surrogates"]
rf = "polarisopt_myplugins.surrogates.random_forest"
```

Users ``pip install polarisopt-myplugins`` and the registries pick up
the new plugins on the next ``polarisopt run``.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Final

from polarisopt.utils.logging import get_logger

log = get_logger(__name__)

#: Entry-point groups we scan. Order is informational.
PLUGIN_GROUPS: Final[tuple[str, ...]] = (
    "polarisopt.designs",
    "polarisopt.surrogates",
    "polarisopt.acquisitions",
    "polarisopt.generators",
    "polarisopt.stops",
    "polarisopt.metrics",
    "polarisopt.simulators",
    "polarisopt.runners",
    "polarisopt.transfers",
)

_LOADED = False


def load_external_plugins(*, force: bool = False) -> list[str]:
    """Import every external plugin module advertised via entry points.

    Idempotent: subsequent calls are no-ops unless ``force=True``.

    Parameters
    ----------
    force : bool, optional
        Re-import every entry-point target even if previously loaded.
        Default ``False`` — once per process is enough.

    Returns
    -------
    list of str
        Module names that were imported (or attempted).
    """
    global _LOADED
    if _LOADED and not force:
        return []
    imported: list[str] = []
    for group in PLUGIN_GROUPS:
        try:
            eps = entry_points(group=group)
        except TypeError:  # pragma: no cover — Python <3.10 compat
            eps = entry_points().get(group, [])  # type: ignore[attr-defined]
        for ep in eps:
            module_name = ep.value
            try:
                importlib.import_module(module_name)
                imported.append(module_name)
                log.debug("Loaded plugin %s from group %s", module_name, group)
            except Exception as exc:  # noqa: BLE001 — plugin load must not crash core
                log.warning(
                    "Failed to import plugin %s (entry point %s in %s): %s",
                    module_name,
                    ep.name,
                    group,
                    exc,
                )
    _LOADED = True
    return imported

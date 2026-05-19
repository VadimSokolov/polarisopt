"""Tests for entry-points-based plugin discovery."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

from polarisopt.utils import plugins


class _FakeEntryPoint:
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value


def test_load_external_plugins_imports_targets() -> None:
    # Create a fake module to "import"
    fake = types.ModuleType("fake_polarisopt_plugin")
    fake.loaded = True   # type: ignore[attr-defined]
    sys.modules["fake_polarisopt_plugin"] = fake

    # Reset the loaded flag
    plugins._LOADED = False

    def fake_entry_points(*, group: str):
        if group == "polarisopt.designs":
            return [_FakeEntryPoint(name="my_grid", value="fake_polarisopt_plugin")]
        return []

    with patch.object(plugins, "entry_points", side_effect=fake_entry_points):
        imported = plugins.load_external_plugins(force=True)

    assert "fake_polarisopt_plugin" in imported


def test_load_external_plugins_is_idempotent() -> None:
    plugins._LOADED = True
    with patch.object(plugins, "entry_points") as m:
        out = plugins.load_external_plugins()
    assert out == []
    assert m.call_count == 0


def test_failing_plugin_does_not_crash() -> None:
    plugins._LOADED = False

    def fake_entry_points(*, group: str):
        if group == "polarisopt.designs":
            return [_FakeEntryPoint(name="bad", value="does.not.exist.module.xyz")]
        return []

    with patch.object(plugins, "entry_points", side_effect=fake_entry_points):
        imported = plugins.load_external_plugins(force=True)

    # We attempted to load it; failure is logged + swallowed.
    # The module name is NOT appended to imported on ImportError.
    assert "does.not.exist.module.xyz" not in imported


def test_plugin_groups_constant_complete() -> None:
    """Sanity: every registry family we have should be in PLUGIN_GROUPS."""
    families = {
        "designs", "surrogates", "acquisitions", "generators",
        "stops", "metrics", "simulators", "runners", "transfers",
    }
    expected = {f"polarisopt.{f}" for f in families}
    assert set(plugins.PLUGIN_GROUPS) == expected

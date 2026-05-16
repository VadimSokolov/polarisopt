from __future__ import annotations

import pytest

from polarisopt.utils.registry import Registry


class _Base:
    pass


def test_register_and_get() -> None:
    r: Registry[_Base] = Registry("widget")

    @r.register("alpha")
    class A(_Base):
        pass

    assert "alpha" in r
    assert r.get("alpha") is A
    assert r.names() == ["alpha"]


def test_unknown_name_lists_known() -> None:
    r: Registry[_Base] = Registry("widget")
    r.register("a")(_Base)
    r.register("b")(_Base)
    with pytest.raises(KeyError, match="Registered: a, b"):
        r.get("c")


def test_duplicate_name_raises() -> None:
    r: Registry[_Base] = Registry("widget")

    @r.register("dupe")
    class _A(_Base):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @r.register("dupe")
        class _B(_Base):
            pass


def test_empty_registry_lookup_message() -> None:
    r: Registry[_Base] = Registry("widget")
    with pytest.raises(KeyError, match="Registered: <none>"):
        r.get("anything")

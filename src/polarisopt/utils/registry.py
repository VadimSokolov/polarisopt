"""Plugin registry — every ABC pluggable from YAML routes through here.

Each pluggable family (designs, surrogates, runners, ...) owns its own
``Registry`` instance. Subclasses register via ``@registry.register("name")``
and YAML lookups go through ``registry.get("name")``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → class registry with a decorator for registration.

    Every pluggable family in polarisopt owns a ``Registry`` instance. The
    decorator pattern lets concrete classes self-register at import time.

    Parameters
    ----------
    family : str
        Human-readable name of the family (e.g. ``"design"``, ``"surrogate"``).
        Used in error messages.

    Examples
    --------
    >>> from abc import ABC, abstractmethod
    >>> class Greeter(ABC):
    ...     @abstractmethod
    ...     def hello(self) -> str: ...
    >>> registry: Registry[Greeter] = Registry("greeter")

    >>> @registry.register("loud")
    ... class LoudGreeter(Greeter):
    ...     def hello(self): return "HELLO!"

    >>> "loud" in registry
    True
    >>> registry.get("loud") is LoudGreeter
    True
    >>> registry.names()
    ['loud']
    """

    def __init__(self, family: str) -> None:
        self._family = family
        self._items: dict[str, type[T]] = {}

    @property
    def family(self) -> str:
        return self._family

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator. ``@registry.register("ei")`` adds the class under that name."""

        def _wrap(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(
                    f"{self._family} '{name}' is already registered to "
                    f"{self._items[name].__qualname__}"
                )
            self._items[name] = cls
            return cls

        return _wrap

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown {self._family} '{name}'. Registered: {available}."
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __len__(self) -> int:
        return len(self._items)

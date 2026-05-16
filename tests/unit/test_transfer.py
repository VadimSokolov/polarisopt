from __future__ import annotations

from pathlib import Path

import pytest

from polarisopt.transfer import LocalTransfer, make_transfer, transfer_registry
from polarisopt.transfer.base import TransferError


def test_local_copy_file(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("hello")
    dst = tmp_path / "sub" / "b.txt"
    LocalTransfer().copy(src, dst)
    assert dst.read_text() == "hello"


def test_local_copy_tree(tmp_path: Path) -> None:
    src = tmp_path / "tree"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("x")
    (src / "sub" / "b.txt").write_text("y")
    dst = tmp_path / "copy"
    LocalTransfer().copy(src, dst, recursive=True)
    assert (dst / "a.txt").read_text() == "x"
    assert (dst / "sub" / "b.txt").read_text() == "y"


def test_local_rejects_dir_without_recursive(tmp_path: Path) -> None:
    src = tmp_path / "tree"
    src.mkdir()
    with pytest.raises(TransferError, match="recursive=True"):
        LocalTransfer().copy(src, tmp_path / "dst")


def test_local_missing_src(tmp_path: Path) -> None:
    with pytest.raises(TransferError, match="source does not exist"):
        LocalTransfer().copy(tmp_path / "nope", tmp_path / "dst")


def test_make_transfer_default_is_local() -> None:
    t = make_transfer(None)
    assert isinstance(t, LocalTransfer)


def test_make_transfer_factory() -> None:
    t = make_transfer({"type": "local"})
    assert isinstance(t, LocalTransfer)


def test_registry_has_local() -> None:
    assert "local" in transfer_registry

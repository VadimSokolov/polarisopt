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


def test_quota_exceeded_subclasses_transfer_error() -> None:
    """Existing ``except TransferError`` callers must still catch
    QuotaExceededError so v0.14 doesn't break backwards-compat.
    """
    from polarisopt.transfer.base import QuotaExceededError, TransferError

    assert issubclass(QuotaExceededError, TransferError)


def test_local_copy_raises_quota_exceeded_on_edquot(
    tmp_path, monkeypatch,
) -> None:
    """EDQUOT (errno 122) from shutil.copy2 → QuotaExceededError, not a
    bare TransferError with a 50-line traceback the master can't classify.
    """
    import errno
    import shutil as _shutil

    from polarisopt.transfer.base import QuotaExceededError
    from polarisopt.transfer.local import LocalTransfer

    src = tmp_path / "src.txt"
    src.write_text("payload")

    def _quota_fail(*_a, **_kw):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(_shutil, "copy2", _quota_fail)
    t = LocalTransfer()
    with pytest.raises(QuotaExceededError, match="EDQUOT"):
        t.copy(src, tmp_path / "dst.txt")


def test_local_copy_raises_quota_exceeded_on_enospc_inside_copytree(
    tmp_path, monkeypatch,
) -> None:
    """ENOSPC raised during copytree (filesystem full) → QuotaExceededError.

    shutil.copytree batches per-file errors into a shutil.Error whose
    args[0] is a list of (src, dst, why) tuples. The detection has to
    walk that structure.
    """
    import errno
    import shutil as _shutil

    from polarisopt.transfer.base import QuotaExceededError
    from polarisopt.transfer.local import LocalTransfer

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x")

    def _enospc_tree(*_a, **_kw):
        raise _shutil.Error([
            (str(src / "a.txt"), str(tmp_path / "dst/a.txt"),
             "[Errno 28] No space left on device: '/dst/a.txt'"),
        ])

    monkeypatch.setattr(_shutil, "copytree", _enospc_tree)
    t = LocalTransfer()
    with pytest.raises(QuotaExceededError, match="ENOSPC"):
        t.copy(src, tmp_path / "dst", recursive=True)
    # Use errno to silence the unused-import linter
    assert errno.ENOSPC == 28

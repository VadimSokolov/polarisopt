"""Tests for the direct Globus-SDK transfer backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polarisopt.transfer import transfer_registry
from polarisopt.transfer.base import TransferError
from polarisopt.transfer.globus import GlobusTransfer


def _gt(tmp_path: Path) -> GlobusTransfer:
    return GlobusTransfer(
        client_id="test-client-id",
        endpoints={
            "/mnt/VMS": "vms-uuid",
            "/lcrc": "lcrc-uuid",
        },
        token_cache=tmp_path / "tokens.json",
        poll_interval=0.0,
    )


def test_registry_has_globus() -> None:
    assert "globus" in transfer_registry


def test_requires_client_id() -> None:
    with pytest.raises(ValueError):
        GlobusTransfer(client_id="", endpoints={"/x": "uuid"})


def test_requires_endpoints() -> None:
    with pytest.raises(ValueError):
        GlobusTransfer(client_id="cid", endpoints={})


def test_endpoint_longest_prefix_match(tmp_path: Path) -> None:
    gt = GlobusTransfer(
        client_id="cid",
        endpoints={
            "/lcrc": "short",
            "/lcrc/project/POLARIS": "long",
        },
    )
    # _endpoint_for is private but we exercise it directly here.
    assert gt._endpoint_for(Path("/lcrc/project/POLARIS/x")) == "long"
    assert gt._endpoint_for(Path("/lcrc/somewhere/else")) == "short"


def test_endpoint_no_match_raises(tmp_path: Path) -> None:
    gt = _gt(tmp_path)
    with pytest.raises(TransferError, match="no Globus endpoint"):
        gt._endpoint_for(Path("/somewhere/else"))


def test_copy_requires_absolute_paths(tmp_path: Path) -> None:
    gt = _gt(tmp_path)
    with pytest.raises(TransferError, match="absolute paths"):
        gt.copy("relative/src", "/lcrc/dst")


def test_copy_submits_and_waits(tmp_path: Path) -> None:
    """Mock globus-sdk entirely and verify submit + poll loop."""
    gt = _gt(tmp_path)
    fake_client = MagicMock()
    fake_client.submit_transfer.return_value = {"task_id": "task-xyz"}
    # First poll returns ACTIVE, second SUCCEEDED — exercise the loop.
    fake_client.get_task.side_effect = [
        {"status": "ACTIVE"},
        {"status": "SUCCEEDED"},
    ]
    gt._client = fake_client

    fake_sdk = MagicMock()
    fake_sdk.TransferData.return_value = MagicMock()

    with patch.object(__import__("polarisopt.transfer.globus", fromlist=["_load_globus"]),
                      "_load_globus", return_value=fake_sdk):
        gt.copy("/mnt/VMS/x", "/lcrc/y", recursive=True)

    fake_client.submit_transfer.assert_called_once()
    assert fake_client.get_task.call_count == 2


def test_copy_raises_on_globus_failure(tmp_path: Path) -> None:
    gt = _gt(tmp_path)
    fake_client = MagicMock()
    fake_client.submit_transfer.return_value = {"task_id": "task-xyz"}
    fake_client.get_task.return_value = {
        "status": "FAILED",
        "nice_status_details": "permission denied",
    }
    gt._client = fake_client

    fake_sdk = MagicMock()
    fake_sdk.TransferData.return_value = MagicMock()

    with patch.object(__import__("polarisopt.transfer.globus", fromlist=["_load_globus"]),
                      "_load_globus", return_value=fake_sdk), pytest.raises(TransferError, match="permission denied"):
        gt.copy("/mnt/VMS/x", "/lcrc/y")


def test_invalid_sync_level() -> None:
    with pytest.raises(ValueError, match="sync_level"):
        GlobusTransfer(
            client_id="cid",
            endpoints={"/x": "uuid"},
            sync_level="not_a_thing",
        )

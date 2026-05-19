"""Direct Globus transfer via ``globus-sdk`` — for non-ANL deployments.

Where :class:`~polarisopt.transfer.anl.AnlTransfer` wraps polaris-studio's
``magic_copy`` (which auto-routes through Globus when paths sit on a
registered endpoint), this transfer talks to the Globus Transfer API
directly. Users register endpoints in their YAML.

Auth model
----------

The simplest model — refresh-token flow with a cached token file —
suits HPC accounts on a shared filesystem. Auth scope is delegated to
the ``globus-sdk`` library; users authenticate once via
``globus login`` (or a one-time browser flow) and the token cache lives
at ``~/.globus/polarisopt/tokens.json``.

For unattended (CI) use, set ``GLOBUS_REFRESH_TOKEN`` in the
environment and we'll skip the interactive bootstrap.

Use this only if you can't use the simpler :class:`AnlTransfer`
(which leverages polarislib's pre-configured endpoints).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from polarisopt.transfer.base import Transfer, TransferError, transfer_registry
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


_DEFAULT_TOKEN_CACHE = Path.home() / ".globus" / "polarisopt" / "tokens.json"
_TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"


def _load_globus():
    """Lazy import of globus-sdk so the module loads cleanly without the extra."""
    try:
        import globus_sdk
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "polarisopt.transfer.globus requires globus-sdk: "
            "pip install 'polarisopt[globus]'"
        ) from exc
    return globus_sdk


@transfer_registry.register("globus")
class GlobusTransfer(Transfer):
    """Direct Globus-SDK-backed file transfer.

    Parameters
    ----------
    client_id : str
        Globus Auth client ID for *your* application registration.
        Required for refresh-token authentication. (Register at
        https://app.globus.org/settings/developers; this is **not**
        polarisopt's responsibility.)
    endpoints : dict[str, str]
        Mapping of path-prefix -> Globus endpoint UUID. The transfer
        looks up which endpoint hosts a given absolute path by matching
        prefixes (longest-prefix wins). Example::

            endpoints:
              /mnt/VMS_DFW: 12345678-...-uuid
              /lcrc/project: abcdef01-...-uuid
    token_cache : path, optional
        Where refresh tokens are cached. Defaults to
        ``~/.globus/polarisopt/tokens.json``.
    sync_level : {"exists", "size", "mtime", "checksum"}, optional
        Globus Transfer "sync level" — what counts as "already there"
        for incremental transfers. Default ``"mtime"``.
    poll_interval : float, optional
        Seconds between Globus Transfer task status polls. Default 5.

    Notes
    -----
    For one-off scripted use, prefer the simpler :class:`AnlTransfer`
    (via the ``[anl]`` extra). This direct-SDK class is for non-ANL
    POLARIS deployments that need to register their own endpoints.

    Examples
    --------
    YAML:

    .. code-block:: yaml

        simulator:
          type: polaris
          options:
            transfer:
              type: globus
              options:
                client_id: 12345678-...-uuid
                endpoints:
                  /mnt/VMS_DFW: aabbccdd-...-uuid
                  /lcrc:        eeff0011-...-uuid
    """

    def __init__(
        self,
        *,
        client_id: str,
        endpoints: dict[str, str],
        token_cache: Path | str | None = None,
        sync_level: str = "mtime",
        poll_interval: float = 5.0,
    ) -> None:
        if not client_id:
            raise ValueError("GlobusTransfer requires client_id")
        if not endpoints:
            raise ValueError("GlobusTransfer requires at least one endpoint mapping")
        if sync_level not in ("exists", "size", "mtime", "checksum"):
            raise ValueError(f"invalid sync_level: {sync_level!r}")
        self.client_id = client_id
        self.endpoints = dict(endpoints)
        self.token_cache = Path(token_cache) if token_cache else _DEFAULT_TOKEN_CACHE
        self.sync_level = sync_level
        self.poll_interval = float(poll_interval)
        self._client: Any = None  # globus_sdk.TransferClient

    # ----- endpoint lookup -----

    def _endpoint_for(self, path: Path) -> str:
        """Resolve which registered endpoint hosts ``path``.

        Longest-prefix match wins. Raises if no prefix matches — local
        copies don't use this transfer.
        """
        abs_path = str(path.resolve()) if path.is_absolute() else str(path)
        matches = [(prefix, uuid) for prefix, uuid in self.endpoints.items() if abs_path.startswith(prefix)]
        if not matches:
            raise TransferError(
                f"no Globus endpoint registered for path {abs_path}. "
                f"Configured prefixes: {sorted(self.endpoints)}"
            )
        # longest prefix wins
        _, uuid = max(matches, key=lambda kv: len(kv[0]))
        return uuid

    # ----- auth -----

    def _client_or_init(self) -> Any:
        if self._client is not None:
            return self._client
        globus_sdk = _load_globus()
        refresh_env = os.environ.get("GLOBUS_REFRESH_TOKEN")
        if refresh_env:
            authorizer = globus_sdk.RefreshTokenAuthorizer(
                refresh_env,
                globus_sdk.NativeAppAuthClient(self.client_id),
            )
        else:
            tokens = self._load_or_acquire_tokens()
            transfer_tokens = tokens["transfer.api.globus.org"]
            authorizer = globus_sdk.RefreshTokenAuthorizer(
                transfer_tokens["refresh_token"],
                globus_sdk.NativeAppAuthClient(self.client_id),
                access_token=transfer_tokens["access_token"],
                expires_at=transfer_tokens["expires_at_seconds"],
            )
        self._client = globus_sdk.TransferClient(authorizer=authorizer)
        return self._client

    def _load_or_acquire_tokens(self) -> dict[str, Any]:
        if self.token_cache.exists():
            return json.loads(self.token_cache.read_text())
        # Interactive bootstrap: user does this once.
        globus_sdk = _load_globus()
        client = globus_sdk.NativeAppAuthClient(self.client_id)
        client.oauth2_start_flow(refresh_tokens=True, requested_scopes=_TRANSFER_SCOPE)
        auth_url = client.oauth2_get_authorize_url()
        print(f"Please log in at: {auth_url}")
        auth_code = input("Paste the resulting auth code: ").strip()
        token_response = client.oauth2_exchange_code_for_tokens(auth_code)
        tokens = token_response.by_resource_server
        self.token_cache.parent.mkdir(parents=True, exist_ok=True)
        self.token_cache.write_text(json.dumps(tokens, indent=2))
        return tokens

    # ----- copy -----

    def copy(self, src: Path | str, dst: Path | str, *, recursive: bool = False) -> None:
        src_p, dst_p = Path(src), Path(dst)
        if not src_p.is_absolute() or not dst_p.is_absolute():
            raise TransferError(
                "GlobusTransfer requires absolute paths on both sides; "
                "got src={src_p}, dst={dst_p}"
            )
        src_ep = self._endpoint_for(src_p)
        dst_ep = self._endpoint_for(dst_p)

        globus_sdk = _load_globus()
        client = self._client_or_init()

        tdata = globus_sdk.TransferData(
            client,
            source_endpoint=src_ep,
            destination_endpoint=dst_ep,
            sync_level=self.sync_level,
        )
        tdata.add_item(str(src_p), str(dst_p), recursive=recursive)

        try:
            submission = client.submit_transfer(tdata)
        except Exception as exc:
            raise TransferError(f"submit_transfer failed: {exc}") from exc

        task_id = submission["task_id"]
        log.info("GlobusTransfer submitted task %s (%s -> %s)", task_id, src_p, dst_p)
        self._wait_for_task(task_id)

    def _wait_for_task(self, task_id: str) -> None:
        client = self._client_or_init()
        while True:
            task = client.get_task(task_id)
            status = task["status"]
            if status == "SUCCEEDED":
                log.info("GlobusTransfer task %s succeeded", task_id)
                return
            if status in ("FAILED", "INACTIVE"):
                raise TransferError(
                    f"Globus task {task_id} ended with status={status}: "
                    f"{task.get('nice_status_details', '')}"
                )
            time.sleep(self.poll_interval)

# How to use Globus for file transfer

ANL POLARIS users store models on the VMS share, reachable only via
Globus from cluster nodes. polarisopt's `transfer.type: anl` backend
wraps polaris-studio's `magic_copy`, which auto-routes through Globus
when either endpoint is a registered Globus path.

## Install the extra

```bash
pip install 'polarisopt[anl]'
```

This pulls in `polaris-studio` for its `magic_copy` helper and Globus
endpoint registry.

## Set up Globus auth once

Follow the polaris-studio Globus authentication guide. Briefly:

```bash
python -c "from polaris.utils.copy_utils import authenticate; authenticate()"
```

You'll be prompted to visit a URL and paste an auth code. The token
caches under `~/.globus_polaris/` and refreshes automatically. Make
sure the dependent scope for the LCRC Improv DTN `data_access` is
included — without it, Globus transfers fail hours into a long job.

## YAML

```yaml
simulator:
  type: polaris
  options:
    binary: /lcrc/project/POLARIS/.../polaris_exe/Integrated_Model.sif
    model_source: /mnt/VMS_DFW/.../DFW_2050_20251028   # ← Globus-backed path
    scenario_file: scenario_abm.json
    output_db_filename: DFW-Result.h5
    num_threads: "16"
    transfer:
      type: anl
      options: {}
```

`magic_copy` introspects `model_source` and `workspace`:

- If either is a registered Globus endpoint (e.g. `/mnt/VMS_*`), it uses
  the Globus Transfer API.
- Otherwise it falls back to local `cp`.

## Limitations

- ANL-only. Other POLARIS deployments need to write their own
  `Transfer` subclass against `globus-sdk` directly (see
  [Plugin authoring](../plugins.md)).
- The first invocation can pause for ~minutes while a fresh Globus
  endpoint connection is negotiated. Subsequent transfers in the same
  process are fast.
- Auth issues bite hours in. Validate with a small test copy
  *before* kicking off a long study.

## Lessons inherited from DOE work

The `DOE_RUNBOOK.md` lessons that apply here:

- **Token + scopes**: refresh tokens before submitting overnight runs.
  Globus scopes can silently expire; a 6-hour calibration that fails
  in the final 30 minutes due to copy-back auth is the worst case.
- **Don't put VMS paths in workspace**: keep the workspace on cluster
  scratch (`/lcrc/...`) for performance; VMS only for read sources and
  final copy-back.
- **Globus's `magic_copy` is single-stream by default**: it's fine for
  model directories (~10–100 GB) but slow for huge files. Consider
  splitting if your model exceeds 200 GB.

## See also

- [AnlTransfer API](../reference/api/transfer/anl.md)
- [Tutorial 05 · First POLARIS run](../tutorials/05-first-polaris.md)

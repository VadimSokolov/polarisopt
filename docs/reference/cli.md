# CLI reference

The ``polarisopt`` command-line tool wraps the Python API. All subcommands
take a study YAML as their first positional argument.

## Global options

```
polarisopt [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] COMMAND ...
```

## Subcommands

### `polarisopt run STUDY`

Execute every phase in ``STUDY``. If the SampleStore already exists at
the workspace, PENDING samples are evaluated before generating new ones
(this is the resume path).

```bash
polarisopt run study.yaml
```

Output: ``completed: <finished>/<total> samples (failed: <failed>)``.

### `polarisopt status STUDY`

Show per-phase counts of pending / running / finished / failed / cancelled
samples for the study referenced by ``STUDY``.

```bash
polarisopt status study.yaml
```

Output example:
```
lhs-screen: {'finished': 16}
bo:         {'finished': 12, 'failed': 1, 'pending': 3}
```

### `polarisopt resume STUDY`

Alias for ``polarisopt run STUDY`` — semantically clearer when picking
up after an interruption. Reads the latest ``phase_state`` checkpoint
for each sequential phase, restores the RNG state, refits the surrogate
from the SampleStore, and continues the loop.

```bash
polarisopt resume study.yaml
```

### `polarisopt cancel STUDY SAMPLE_ID`

Cancel one sample by id. The underlying Slurm job is ``scancel``'d (if
any) and the SampleStore row transitions to CANCELLED.

```bash
polarisopt cancel study.yaml 42
```

Idempotent — already-terminal samples return unchanged.

### `polarisopt abort STUDY`

Cancel every non-terminal sample in the study. Useful for emergency
stops without killing the master process.

```bash
polarisopt abort study.yaml
```

Output: ``aborted <count> non-terminal sample(s)``.

### `polarisopt logs STUDY SAMPLE_ID [-f] [-n N]`

Print every ``*.log``, ``*.out``, ``*.err`` file in ``sample.folder``.

| Flag | Meaning |
|---|---|
| ``-f``, ``--follow`` | Stream new lines from the largest file (``tail -f`` semantics). Ctrl-C to exit. |
| ``-n N``, ``--lines N`` | Print only the last ``N`` lines of each file. |

```bash
polarisopt logs study.yaml 42
polarisopt logs study.yaml 42 --follow
polarisopt logs study.yaml 42 -n 100
```

### `polarisopt retry-failed STUDY [--id N]... [--run]`

Flip FAILED samples back to PENDING. A subsequent ``polarisopt run`` /
``resume`` will pick them up and re-evaluate.

| Flag | Meaning |
|---|---|
| ``--id N`` | Retry only a specific sample id. Repeat for many. Default: every FAILED sample. |
| ``--run`` | Immediately re-run the study after flipping (one fewer command to type). |

```bash
polarisopt retry-failed study.yaml                 # retry every FAILED
polarisopt retry-failed study.yaml --id 42         # retry sample 42 only
polarisopt retry-failed study.yaml --id 42 --id 47 # retry several
polarisopt retry-failed study.yaml --run           # retry + re-run in one command
```

### `polarisopt examples list | show NAME | copy NAME DEST`

Bundled example study YAMLs.

```bash
polarisopt examples list                           # branin, morris, multi-objective, polaris-slurm
polarisopt examples show branin                    # cat the YAML
polarisopt examples copy branin ./my-study.yaml    # copy for local editing
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Click error (bad argument, missing file, etc.) |
| 2 | Click usage error |

Subcommands raise `click.ClickException` for user-facing failures
(missing store, unknown sample id) — these print to stderr and exit 1.

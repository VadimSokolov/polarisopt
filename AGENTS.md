# AGENTS.md — guidance for AI coding agents

If you're an AI agent (Claude, Cursor, Copilot, etc.) contributing to
this codebase, read this first. It's a condensed companion to the
[full docs](https://anl-polaris.github.io/polaris-hpc/) tuned for the
kind of context an LLM needs to be effective here.

> **Driving polarisopt from another agent's session, not modifying it?**
> Read [docs/operating-as-an-agent.md](docs/operating-as-an-agent.md)
> instead. This file is for agents modifying polarisopt source; the
> other file is for agents *using* polarisopt to drive POLARIS studies.
> The two roles want different things.

## What is polarisopt

polarisopt is a Python library for **design-of-experiments and Bayesian
optimization** that targets the [POLARIS](https://polaris.taps.anl.gov/)
agent-based transportation simulator. It runs on Slurm clusters via a
master/slave architecture: a lightweight master Python process holds
study state in a SQLite-backed SampleStore and submits one sbatch job
per sample evaluation.

Every swappable piece — designs, surrogates, acquisition functions,
sample generators, stopping criteria, metrics, simulators, runners,
file-transfer backends — is an ABC backed by a plugin Registry. YAML
study files reference plugins by string name.

## Project conventions

### Python

- **Python 3.11+** required. Use `|`, `Self`, `match`, modern typing.
- **NumPy-style docstrings** everywhere public. Sections: Parameters,
  Returns, Raises, Examples, See Also, Notes. mkdocstrings renders
  these into the API docs.
- **Type hints** on every public API. `from __future__ import annotations`
  is fine for forward refs. Pydantic + dataclasses preferred over
  hand-rolled validation.
- **`logging.getLogger(__name__)`**, never `print()`, in library code.
  The CLI configures handlers; library code stays quiet.
- **PEP-8 with two carveouts** (see `pyproject.toml ruff per-file-ignores`):
  uppercase `X` / `Y` are fine in ML modules; tests can import after
  `pytest.importorskip("torch")` (E402).

### Code organization

- **One ABC per package**: `polarisopt/design/base.py`, `surrogates/base.py`,
  etc. Concrete plugins live as siblings (e.g. `design/lhs.py`).
- **Registries are package-level**: `design_registry`, `surrogate_registry`,
  ... importing the package's `__init__.py` registers all built-ins.
- **Optional dependencies are lazy-imported** inside the modules that
  need them. Always inside a try/except ImportError that raises a
  clear error message naming the extra (`[bo]`, `[anl]`).
- **Per-family `make_xxx(spec)` factory** wraps `registry.get(spec["type"])`
  + `cls(**spec.get("options", {}))`. This is what YAML loaders call.

### Tests

- `tests/unit/` for fast, isolated tests. `tests/integration/` for
  end-to-end YAML→Study→SampleStore flows.
- **No real Slurm cluster needed**. Tests use a `FakeShell` (see
  `tests/conftest.py::FakeShell`) injected into `SlurmRunner` to
  programmatically respond to `sbatch`/`squeue`/`sacct`/`scancel`.
- **BoTorch tests** gate via `pytest.importorskip("torch")` at the top
  of the file. Tests that don't need BO skip cleanly.
- **Docstring examples** run via `pytest --doctest-modules src/polarisopt`.
  Use `# doctest: +SKIP` for examples that need real paths / clusters.

### Commits

- **One feature per commit**. Verbose commit messages with rationale,
  not just "Add foo".
- **Do not add AI-agent co-author tags.** Commit as the human author
  on whose behalf you're working. The git author/committer is enough.
- Use a HEREDOC for multi-line messages so formatting survives.

## How to add a plugin (checklist)

1. Pick the right ABC (`Design`, `Surrogate`, `Acquisition`,
   `SampleGenerator`, `StoppingCriterion`, `Metric`, `Simulator`,
   `Runner`, `Transfer`).
2. Subclass it under `src/polarisopt/<family>/<short_name>.py`.
3. Decorate with `@<family>_registry.register("plugin_name")`.
4. Implement the abstract methods. Use NumPy-style docstrings with
   `Examples`.
5. Add unit tests in `tests/unit/test_<family>_<name>.py`.
6. Re-export from `src/polarisopt/<family>/__init__.py` if it's part
   of the public API.
7. Run `pytest -q` and `ruff check src tests`. Both must pass.

A full worked example is in [Tutorial 06 · Writing a plugin](docs/tutorials/06-write-a-plugin.md).

## How to run tests

```bash
# Set up once
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,bo]"

# Fast test pass
pytest -q

# Including doctest examples
pytest --doctest-modules src/polarisopt -q

# Just one file
pytest tests/unit/test_design.py -q

# With logging
pytest -q --log-cli-level=INFO

# Lint
ruff check src tests
```

## Where things live (1-line per directory)

| Path | Role |
|---|---|
| `src/polarisopt/parameters/` | `ParameterSpace`, POLARIS JSON injection |
| `src/polarisopt/samples/` | `Sample` dataclass + SQLite-backed `SampleStore` |
| `src/polarisopt/config/` | pydantic study YAML schema + Jinja2 templating |
| `src/polarisopt/design/` | Static DOE: LHS, Morris, Sobol, manual |
| `src/polarisopt/surrogates/` | `Surrogate` ABC + BoTorch GP |
| `src/polarisopt/acquisition/` | LogEI, qLogEI, qLogEHVI (via BoTorch) |
| `src/polarisopt/generators/` | `SampleGenerator` (batch-first) |
| `src/polarisopt/stop/` | Stopping criteria + any/all combinators |
| `src/polarisopt/metrics/` | `Metric` ABC + identity, link_moe, choice_share |
| `src/polarisopt/simulator/` | `Simulator` ABC + mock + polaris |
| `src/polarisopt/runners/` | `Runner` ABC + local + slurm |
| `src/polarisopt/transfer/` | `Transfer` ABC + local + anl (Globus) |
| `src/polarisopt/studies/` | Orchestrators: static, sequential, runner, ops |
| `src/polarisopt/utils/` | logging, paths, registry |
| `src/polarisopt/cli.py` | `polarisopt run|status|resume|cancel|abort|logs` |
| `src/polarisopt/eqsql_compat.py` | Drop-in shim for `polaris.hpc.eqsql` |
| `tests/unit/` | Per-module tests |
| `tests/integration/` | End-to-end YAML→run tests |
| `docs/` | mkdocs site (this file + tutorials, how-to, concepts, reference) |

## Public API surface (stable, OK to depend on)

```python
# Top-level
import polarisopt
polarisopt.__version__

# Domain primitives
from polarisopt.parameters import Parameter, ParameterSpace, ParameterType
from polarisopt.samples import Sample, SampleStatus, SampleStore

# Configuration
from polarisopt.config import StudyConfig, load_study_config, render_yaml

# Orchestration
from polarisopt.studies import StudyRunner, StaticDesignStudy, SequentialDesignStudy
from polarisopt.studies import cancel_sample, abort_study, sample_log_paths

# Pluggable families (each module exposes the ABC, concrete builtins, factory, registry)
from polarisopt.design     import Design, make_design, design_registry
from polarisopt.surrogates import Surrogate, make_surrogate, surrogate_registry
from polarisopt.acquisition import AcquisitionFunction, make_acquisition, acquisition_registry
from polarisopt.generators import SampleGenerator, GeneratorContext, make_generator, generator_registry
from polarisopt.stop       import StoppingCriterion, make_stop, stop_registry
from polarisopt.metrics    import Metric, make_metric, metric_registry
from polarisopt.simulator  import Simulator, make_simulator, simulator_registry
from polarisopt.runners    import Runner, JobSpec, Job, JobStatus
from polarisopt.transfer   import Transfer, make_transfer, transfer_registry

# Compat
from polarisopt import eqsql_compat
```

Anything starting with `_` is internal. Don't import directly.

## Don't do these

- ❌ Don't import `polarislib`, `torch`, `botorch`, `gpytorch`, or
  `globus-sdk` at the top of any module **outside** the designated
  modules. Use lazy `try/except ImportError` inside the relevant
  function, and raise a clear error naming the missing extra.
- ❌ Don't `print()` in library code — use `logging.getLogger(__name__)`.
- ❌ Don't mutate a `Sample` without calling `store.update(sample)`. The
  store is the source of truth, in-memory mutations are lost on resume.
- ❌ Don't catch exceptions silently. Either re-raise with context or
  log + transition the sample to FAILED with a `message`.
- ❌ Don't write to files outside `workspace`. The whole point of the
  per-study workspace is that wiping it cleans up the study.
- ❌ Don't add `__init__.py` autoload of optional deps without the
  try/except ImportError + lazy pattern (see
  `src/polarisopt/surrogates/__init__.py` and `acquisition/__init__.py`
  for the template).
- ❌ Don't break the master/slave boundary: master code must not
  execute POLARIS-binary calls. All execution goes through `Runner`.

## Style red flags to call out in review

- Setting `os.chdir` anywhere in library code — breaks parallel
  `LocalRunner` use.
- `pickle` on user-controlled data — security risk.
- Numpy `np.random` global state instead of explicit `Generator`.
- Hard-coded paths in tests (use `tmp_path`).
- Bare `except:` or `except Exception:` without log+re-raise.

## Useful entry points

- The CLI: [`src/polarisopt/cli.py`](src/polarisopt/cli.py)
- The orchestrator master loop: [`src/polarisopt/studies/base.py::_evaluate_batch`](src/polarisopt/studies/base.py)
- The sequential BO loop: [`src/polarisopt/studies/sequential.py::SequentialDesignStudy.run`](src/polarisopt/studies/sequential.py)
- The Slurm runner: [`src/polarisopt/runners/slurm.py`](src/polarisopt/runners/slurm.py)
- The POLARIS simulator: [`src/polarisopt/simulator/polaris.py`](src/polarisopt/simulator/polaris.py)

When in doubt, start with the test file for the module you're editing —
it'll show you the expected interaction.

## See also

- [Full documentation](https://anl-polaris.github.io/polaris-hpc/)
- [`docs/llms.txt`](docs/llms-txt.md) — compact summary in flat plain text for LLM context budgets

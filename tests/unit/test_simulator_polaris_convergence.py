"""Unit tests for PolarisConvergenceSimulator."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.samples.sample import Sample
from polarisopt.simulator import PolarisConvergenceSimulator
from polarisopt.simulator.base import SimulatorError


def _build_fake_model(root: Path, scenario_name: str = "scenario_abm.json") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "DestinationChoice.json").write_text(
        json.dumps({"Hard": {"trip_threshold": 0.0}})
    )
    (root / scenario_name).write_text(
        json.dumps(
            {
                "Output controls": {"output_dir_name": "DFW"},
                "General simulation controls": {"database_name": "DFW"},
            }
        )
    )
    return root


@pytest.fixture
def space() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [Parameter("trip_threshold", "DestinationChoice.json", 0.0, 1.0)]
    )


@pytest.fixture
def runner_script(tmp_path: Path) -> Path:
    p = tmp_path / "runner.py"
    p.write_text("# placeholder runner\nimport sys\n")
    return p


def _make_sim(tmp_path: Path, runner_script: Path, **extras) -> PolarisConvergenceSimulator:
    model = _build_fake_model(tmp_path / "src_model")
    return PolarisConvergenceSimulator(
        runner_script=str(runner_script),
        binary="/usr/bin/echo",  # placeholder; never invoked directly
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="DFW-Demand.sqlite",
        output_dir_key=("Output controls", "output_dir_name"),
        num_threads="4",
        **extras,
    )


def test_init_rejects_unknown_iteration_type(tmp_path: Path, runner_script: Path) -> None:
    with pytest.raises(SimulatorError, match="iteration_type"):
        _make_sim(tmp_path, runner_script, iteration_type="dance")


def test_init_rejects_missing_runner_script(tmp_path: Path) -> None:
    with pytest.raises(SimulatorError, match="runner_script not found"):
        _make_sim(tmp_path, runner_script=tmp_path / "no-such.py")


def test_prepare_builds_runner_invocation(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    sim = _make_sim(
        tmp_path,
        runner_script,
        runner_options={
            "population_scale_factor": 0.05,
            "num_abm_runs": 1,
            "do_skim": False,
        },
        setup_commands=["module purge", "module load singularity"],
    )
    sample = Sample(id=1, inputs=np.array([0.42]))
    workspace = tmp_path / "experiments" / "sim-1"
    spec = sim.prepare(sample, space, workspace)
    assert "module purge" in spec.command
    assert "module load singularity" in spec.command
    assert "--threads=4" in spec.command
    assert "--population-scale-factor=0.05" in spec.command
    assert "--num-abm-runs=1" in spec.command
    assert "--do-skim=false" in spec.command  # bool rendering
    # parameter was injected
    payload = json.loads((workspace / "DestinationChoice.json").read_text())
    assert payload["Hard"]["trip_threshold"] == pytest.approx(0.42)
    assert spec.env["POLARIS_NUM_THREADS"] == "4"


def test_prepare_extra_env_passes_through(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    sim = _make_sim(
        tmp_path,
        runner_script,
        env={"LD_LIBRARY_PATH": "/foo/lib", "MOD_SPATIALITE_PATH": "/foo/mod"},
    )
    sample = Sample(id=2, inputs=np.array([0.5]))
    spec = sim.prepare(sample, space, tmp_path / "sim-2")
    assert spec.env["LD_LIBRARY_PATH"] == "/foo/lib"
    assert spec.env["MOD_SPATIALITE_PATH"] == "/foo/mod"
    assert spec.env["POLARIS_NUM_THREADS"] == "4"


def test_prepare_shell_escapes_runner_options(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    """runner_options values with spaces / shell metacharacters must survive
    intact through the rendered shell command (CodeRabbit finding).
    """
    sim = _make_sim(
        tmp_path,
        runner_script,
        runner_options={
            "tag": "needs spaces and $vars",
            "shell_special": "a;b|c&d",
        },
    )
    sample = Sample(id=42, inputs=np.array([0.5]))
    workspace = tmp_path / "sim-42"
    spec = sim.prepare(sample, space, workspace)
    # The values are quoted as single shell tokens.
    assert "--tag='needs spaces and $vars'" in spec.command
    assert "--shell-special='a;b|c&d'" in spec.command
    # And NOT present unquoted (which would shell-split).
    assert "--tag=needs spaces" not in spec.command


def test_resolve_output_dir_polarislib_naming(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    sim = _make_sim(tmp_path, runner_script, iteration_type="abm_init")
    sample = Sample(id=3, inputs=np.array([0.3]))
    workspace = tmp_path / "sim-3"
    sim.prepare(sample, space, workspace)
    # polarislib writes DFW_01_abm_init_iteration_<N>
    for n in (0, 1, 2):
        d = workspace / f"DFW_01_abm_init_iteration_{n}"
        d.mkdir(parents=True, exist_ok=True)
        # collect_output also expects the result_db file
        with h5py.File(d / "DFW-Demand.sqlite", "w") as f:
            f.create_group("anything")
    sample.folder = workspace
    out = sim.collect_output(sample)
    assert out["iteration"] == 2
    assert out["output_dir"].endswith("DFW_01_abm_init_iteration_2")
    # demand_db alias is exposed
    assert out["demand_db"] == out["result_path"]


def test_resolve_output_dir_unnumbered_polarislib_naming(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    """When polarislib's iteration_number is None it omits the _<N> suffix."""
    sim = _make_sim(tmp_path, runner_script, iteration_type="abm_init")
    sample = Sample(id=33, inputs=np.array([0.3]))
    workspace = tmp_path / "sim-33"
    sim.prepare(sample, space, workspace)
    d = workspace / "DFW_01_abm_init_iteration"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "DFW-Demand.sqlite", "w") as f:
        f.create_group("anything")
    sample.folder = workspace
    out = sim.collect_output(sample)
    # Unsuffixed = baseline = iteration 0 (so IdentityMetric on iteration
    # works on baselines without special-casing None).
    assert out["iteration"] == 0
    assert out["output_dir"].endswith("DFW_01_abm_init_iteration")


def test_resolve_output_dir_prefers_numbered_over_unnumbered(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    """If both numbered and unnumbered dirs exist, the highest-numbered wins."""
    sim = _make_sim(tmp_path, runner_script, iteration_type="abm_init")
    sample = Sample(id=44, inputs=np.array([0.3]))
    workspace = tmp_path / "sim-44"
    sim.prepare(sample, space, workspace)
    for name in (
        "DFW_01_abm_init_iteration",
        "DFW_01_abm_init_iteration_0",
        "DFW_01_abm_init_iteration_1",
    ):
        d = workspace / name
        d.mkdir(parents=True, exist_ok=True)
        with h5py.File(d / "DFW-Demand.sqlite", "w") as f:
            f.create_group("anything")
    sample.folder = workspace
    out = sim.collect_output(sample)
    assert out["output_dir"].endswith("_iteration_1")
    assert out["iteration"] == 1


def test_resolve_output_dir_no_match_raises(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    sim = _make_sim(tmp_path, runner_script)
    sample = Sample(id=4, inputs=np.array([0.1]))
    workspace = tmp_path / "sim-4"
    sim.prepare(sample, space, workspace)
    sample.folder = workspace
    with pytest.raises(SimulatorError, match="no output directory matching"):
        sim.collect_output(sample)


def test_make_simulator_factory(tmp_path: Path, runner_script: Path) -> None:
    """Round-trip through simulator_registry / make_simulator."""
    from polarisopt.simulator import make_simulator

    model = _build_fake_model(tmp_path / "m")
    sim = make_simulator(
        {
            "type": "polaris_convergence",
            "options": {
                "runner_script": str(runner_script),
                "binary": "/usr/bin/echo",
                "model_source": str(model),
                "scenario_file": "scenario_abm.json",
                "output_db_filename": "DFW-Demand.sqlite",
                "output_dir_key": ["Output controls", "output_dir_name"],
            },
        }
    )
    assert isinstance(sim, PolarisConvergenceSimulator)


def test_default_output_dir_key_is_polarislib_correct(
    tmp_path: Path, runner_script: Path
) -> None:
    """polarislib scenarios use ``output_dir_name`` not ``output_directory``."""
    model = _build_fake_model(tmp_path / "m2")
    sim = PolarisConvergenceSimulator(
        runner_script=str(runner_script),
        binary="/usr/bin/echo",
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="DFW-Demand.sqlite",
    )
    assert sim.output_dir_key == ("Output controls", "output_dir_name")


def test_collect_output_exposes_progress_log_path(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    sim = _make_sim(tmp_path, runner_script, iteration_type="abm_init")
    sample = Sample(id=77, inputs=np.array([0.3]))
    workspace = tmp_path / "sim-77"
    sim.prepare(sample, space, workspace)
    out_dir = workspace / "DFW_01_abm_init_iteration_3"
    out_dir.mkdir(parents=True)
    with h5py.File(out_dir / "DFW-Demand.sqlite", "w") as f:
        f.create_group("anything")
    log_dir = out_dir / "log"
    log_dir.mkdir()
    progress = log_dir / "polaris_progress.log"
    progress.write_text("sim hour 10\n")
    sample.folder = workspace

    out = sim.collect_output(sample)
    assert out["progress_log_path"] == str(progress)


def test_collect_output_progress_log_path_is_none_when_missing(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    sim = _make_sim(tmp_path, runner_script, iteration_type="abm_init")
    sample = Sample(id=88, inputs=np.array([0.3]))
    workspace = tmp_path / "sim-88"
    sim.prepare(sample, space, workspace)
    out_dir = workspace / "DFW_01_abm_init_iteration_1"
    out_dir.mkdir(parents=True)
    with h5py.File(out_dir / "DFW-Demand.sqlite", "w") as f:
        f.create_group("anything")
    sample.folder = workspace

    out = sim.collect_output(sample)
    assert out["progress_log_path"] is None


def test_single_iteration_injects_zero_runs(
    tmp_path: Path, runner_script: Path
) -> None:
    sim = _make_sim(tmp_path, runner_script, single_iteration=True)
    assert sim.single_iteration is True
    assert sim.runner_options["num_abm_runs"] == 0
    assert sim.runner_options["num_dta_runs"] == 0


def test_single_iteration_rejects_conflicting_runner_options(
    tmp_path: Path, runner_script: Path
) -> None:
    with pytest.raises(SimulatorError, match="single_iteration=True forces"):
        _make_sim(
            tmp_path,
            runner_script,
            single_iteration=True,
            runner_options={"num_abm_runs": 3},
        )


def test_single_iteration_accepts_explicit_zero(
    tmp_path: Path, runner_script: Path
) -> None:
    """Setting num_abm_runs=0 explicitly with single_iteration=True is fine."""
    sim = _make_sim(
        tmp_path,
        runner_script,
        single_iteration=True,
        runner_options={"num_abm_runs": 0, "num_dta_runs": 0, "population_scale_factor": 0.05},
    )
    assert sim.runner_options["num_abm_runs"] == 0
    assert sim.runner_options["population_scale_factor"] == 0.05


def test_single_iteration_assertion_catches_extra_iteration_dir(
    tmp_path: Path, space: ParameterSpace, runner_script: Path
) -> None:
    """If a normal_iteration dir slips past, collect_output should raise."""
    sim = _make_sim(
        tmp_path, runner_script, iteration_type="abm_init", single_iteration=True,
    )
    sample = Sample(id=99, inputs=np.array([0.3]))
    workspace = tmp_path / "sim-99"
    sim.prepare(sample, space, workspace)
    # Expected abm_init dir + UNexpected dta iteration dir.
    abm = workspace / "DFW_01_abm_init_iteration_0"
    abm.mkdir()
    with h5py.File(abm / "DFW-Demand.sqlite", "w") as f:
        f.create_group("anything")
    extra = workspace / "DFW_dta_iteration_1"
    extra.mkdir()
    sample.folder = workspace

    with pytest.raises(SimulatorError, match="extra iteration dir"):
        sim.collect_output(sample)


def test_disable_async_callback_default_on(
    tmp_path: Path, runner_script: Path
) -> None:
    """Default is True (preserve per-iteration artifacts)."""
    sim = _make_sim(tmp_path, runner_script)
    assert sim.disable_async_callback is True
    assert sim.runner_options["disable_async_callback"] is True


def test_disable_async_callback_can_be_overridden(
    tmp_path: Path, runner_script: Path
) -> None:
    sim = _make_sim(tmp_path, runner_script, disable_async_callback=False)
    assert sim.disable_async_callback is False
    assert sim.runner_options["disable_async_callback"] is False


def test_disable_async_callback_in_runner_options_wins(
    tmp_path: Path, runner_script: Path
) -> None:
    """Explicit runner_options entry overrides the kwarg default."""
    sim = _make_sim(
        tmp_path, runner_script,
        runner_options={"disable_async_callback": False},
    )
    # The kwarg defaults to True but the explicit runner_options entry wins.
    assert sim.runner_options["disable_async_callback"] is False


def test_disable_async_callback_in_known_runner_options(
    tmp_path: Path, runner_script: Path
) -> None:
    sim = _make_sim(tmp_path, runner_script)
    assert "disable_async_callback" in sim.KNOWN_RUNNER_OPTIONS
    # And so it's never flagged as unknown:
    assert "disable_async_callback" not in sim.unknown_runner_options()


def test_unknown_runner_options_flags_typos(
    tmp_path: Path, runner_script: Path
) -> None:
    sim = _make_sim(
        tmp_path,
        runner_script,
        runner_options={
            "population_scale_factor": 0.05,   # known
            "num_abm_runs": 1,                  # known
            "population_scal_factor": 0.05,    # typo!
            "completely_made_up_knob": "foo",  # branch-specific or invented
        },
    )
    unknown = sim.unknown_runner_options()
    assert "population_scal_factor" in unknown
    assert "completely_made_up_knob" in unknown
    assert "population_scale_factor" not in unknown
    assert "num_abm_runs" not in unknown

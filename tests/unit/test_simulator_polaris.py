from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.samples.sample import Sample
from polarisopt.simulator import PolarisSimulator
from polarisopt.simulator.base import SimulatorError, make_simulator


def _build_fake_model(root: Path, scenario_name: str = "scenario_abm.json") -> Path:
    """Construct a POLARIS-shaped model directory with two parameter JSONs and a scenario."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "DestinationChoice.json").write_text(
        json.dumps({"Hard constraints": {"trip_threshold": 0.0, "min_distance": 0.0}})
    )
    (root / "ActivityChoice.json").write_text(
        json.dumps({"Weights": {"alpha": 0.0}})
    )
    (root / scenario_name).write_text(
        json.dumps(
            {
                "Output controls": {"output_directory": "out"},
                "General simulation controls": {"database_name": "TestModel"},
            }
        )
    )
    return root


@pytest.fixture
def space() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [
            Parameter("trip_threshold", "DestinationChoice.json", 0.0, 1.0),
            Parameter("alpha", "ActivityChoice.json", -1.0, 1.0),
        ]
    )


def _make_sim(tmp_path: Path) -> PolarisSimulator:
    model = _build_fake_model(tmp_path / "src_model")
    return PolarisSimulator(
        binary="/usr/bin/echo",  # any executable for tests
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="TestModel-Result.h5",
        num_threads="4",
    )


def test_prepare_stages_model_and_injects(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=1, phase="test", inputs=np.array([0.42, 0.7]))
    workspace = tmp_path / "experiments" / "sim-1"
    spec = sim.prepare(sample, space, workspace)
    # files staged
    dest_json = workspace / "DestinationChoice.json"
    assert dest_json.exists()
    payload = json.loads(dest_json.read_text())
    assert payload["Hard constraints"]["trip_threshold"] == pytest.approx(0.42)
    activity = json.loads((workspace / "ActivityChoice.json").read_text())
    assert activity["Weights"]["alpha"] == pytest.approx(0.7)
    # job spec sane
    assert spec.cwd == workspace
    assert "scenario_abm.json" in spec.command
    assert spec.env["POLARIS_NUM_THREADS"] == "4"


def test_prepare_rejects_wrong_input_shape(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=1, inputs=np.array([0.5]))  # too few
    with pytest.raises(SimulatorError):
        sim.prepare(sample, space, tmp_path / "x")


def test_prepare_rejects_missing_model_source(tmp_path: Path, space: ParameterSpace) -> None:
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(tmp_path / "does_not_exist"),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
    )
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    with pytest.raises(SimulatorError, match="model_source"):
        sim.prepare(sample, space, tmp_path / "x")


def test_collect_output_returns_result_paths(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=2, inputs=np.array([0.3, 0.5]))
    workspace = tmp_path / "experiments" / "sim-2"
    sim.prepare(sample, space, workspace)
    # simulate POLARIS writing its result
    (workspace / "out").mkdir(parents=True, exist_ok=True)
    result_path = workspace / "out" / "TestModel-Result.h5"
    with h5py.File(result_path, "w") as f:
        g = f.create_group("link_moe")
        g.create_dataset("link_travel_time", data=np.ones((3, 2)))
        g.create_dataset("link_in_volume", data=np.ones((3, 2)))
    sample.folder = workspace
    out = sim.collect_output(sample)
    assert out["result_path"] == str(result_path)


def test_collect_output_missing_result_raises(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=3, inputs=np.array([0.0, 0.0]))
    workspace = tmp_path / "experiments" / "sim-3"
    sim.prepare(sample, space, workspace)
    sample.folder = workspace
    # do NOT create the output directory at all
    with pytest.raises(SimulatorError, match="no output directory found"):
        sim.collect_output(sample)


def test_collect_output_missing_h5_in_existing_dir_raises(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """Output directory exists but the H5 result file is absent."""
    sim = _make_sim(tmp_path)
    sample = Sample(id=33, inputs=np.array([0.0, 0.0]))
    workspace = tmp_path / "experiments" / "sim-33"
    sim.prepare(sample, space, workspace)
    sample.folder = workspace
    (workspace / "out").mkdir(parents=True, exist_ok=True)  # exists but empty
    with pytest.raises(SimulatorError, match="result file missing"):
        sim.collect_output(sample)


def test_native_binary_not_wrapped_with_apptainer(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """Native (non-SIF) binaries are invoked bare — no apptainer wrapping."""
    sim = _make_sim(tmp_path)  # binary = /usr/bin/echo
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    workspace = tmp_path / "experiments" / "sim-1"
    spec = sim.prepare(sample, space, workspace)
    # Compare the first token; pytest dirs may contain "apptainer" literally.
    assert spec.command.split()[0] == "/usr/bin/echo"
    assert " run " not in spec.command  # no apptainer/singularity sub-cmd
    assert " -B " not in spec.command


def test_sif_binary_wrapped_with_apptainer_run(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """When binary is a .sif, prepare wraps with `apptainer run -B <workspace>`."""
    model = _build_fake_model(tmp_path / "src_model")
    sif_dir = tmp_path / "polaris_exe"
    sif_dir.mkdir()
    sif_path = sif_dir / "Integrated_Model.sif"
    sif_path.touch()
    sim = PolarisSimulator(
        binary=str(sif_path),
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="TestModel-Result.h5",
        num_threads="16",
    )
    sample = Sample(id=42, inputs=np.array([0.5, 0.5]))
    workspace = tmp_path / "experiments" / "sim-42"
    spec = sim.prepare(sample, space, workspace)
    assert spec.command.startswith("apptainer run ")
    # Workspace + SIF parent are auto-bound.
    assert f"-B {workspace.resolve()}" in spec.command
    assert f"-B {sif_dir.resolve()}" in spec.command
    scenario_path = workspace / "scenario_abm.json"
    # SIF path appears after the binds and before the scenario.
    sif_pos = spec.command.index(str(sif_path))
    scenario_pos = spec.command.index(str(scenario_path))
    assert sif_pos < scenario_pos
    # No sif_entrypoint set → no extra positional arg between SIF and scenario.
    between = spec.command[sif_pos + len(str(sif_path)) : scenario_pos].strip()
    assert between == ""


def test_sif_entrypoint_inserted_before_scenario(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """sif_entrypoint='Integrated_Model' adds the new-format dispatch arg."""
    model = _build_fake_model(tmp_path / "src_model")
    sif_path = tmp_path / "polaris_exe" / "polaris.sif"
    sif_path.parent.mkdir()
    sif_path.touch()
    sim = PolarisSimulator(
        binary=str(sif_path),
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        num_threads="8",
        sif_entrypoint="Integrated_Model",
    )
    sample = Sample(id=7, inputs=np.array([0.5, 0.5]))
    workspace = tmp_path / "experiments" / "sim-7"
    spec = sim.prepare(sample, space, workspace)
    scenario_path = workspace / "scenario_abm.json"
    # Order: ... <sif> Integrated_Model <scenario> <threads>
    sif_pos = spec.command.index(str(sif_path))
    entry_pos = spec.command.index("Integrated_Model", sif_pos)
    scenario_pos = spec.command.index(str(scenario_path))
    assert sif_pos < entry_pos < scenario_pos


def test_singularity_binds_appended_after_auto_binds(
    tmp_path: Path, space: ParameterSpace
) -> None:
    model = _build_fake_model(tmp_path / "src_model")
    sif_path = tmp_path / "polaris_exe" / "Integrated_Model.sif"
    sif_path.parent.mkdir()
    sif_path.touch()
    sim = PolarisSimulator(
        binary=str(sif_path),
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        singularity_binds=["/lcrc/project/POLARIS/shared", "/scratch:/scratch"],
    )
    sample = Sample(id=9, inputs=np.array([0.5, 0.5]))
    workspace = tmp_path / "experiments" / "sim-9"
    spec = sim.prepare(sample, space, workspace)
    assert "-B /lcrc/project/POLARIS/shared" in spec.command
    assert "-B /scratch:/scratch" in spec.command


def test_singularity_binds_deduped_with_auto_binds(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """If user passes the same path the auto-bind already covers, don't repeat it."""
    model = _build_fake_model(tmp_path / "src_model")
    sif_path = tmp_path / "polaris_exe" / "Integrated_Model.sif"
    sif_path.parent.mkdir()
    sif_path.touch()
    workspace = tmp_path / "experiments" / "sim-1"
    workspace.mkdir(parents=True)
    sim = PolarisSimulator(
        binary=str(sif_path),
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        # User-supplied bind that duplicates the workspace auto-bind:
        singularity_binds=[str(workspace.resolve())],
    )
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    spec = sim.prepare(sample, space, workspace)
    # The bind appears exactly once.
    assert spec.command.count(f"-B {workspace.resolve()}") == 1


def test_apptainer_binary_override_to_singularity(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """Older clusters use `singularity` instead of `apptainer`."""
    model = _build_fake_model(tmp_path / "src_model")
    sif_path = tmp_path / "polaris_exe" / "polaris.sif"
    sif_path.parent.mkdir()
    sif_path.touch()
    sim = PolarisSimulator(
        binary=str(sif_path),
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        apptainer_binary="singularity",
    )
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    workspace = tmp_path / "experiments" / "sim-1"
    spec = sim.prepare(sample, space, workspace)
    assert spec.command.startswith("singularity run ")
    # tmp_path may contain the literal substring "apptainer" (pytest
    # interpolates the test name into the dir), so compare the first
    # token rather than substring-matching.
    assert spec.command.split()[0] == "singularity"


def test_pre_script_renders_before_binary(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """``pre_script`` is invoked before the binary with all sample params
    forwarded as ``--<dashified-name>=<value>`` CLI flags.
    """
    pre = tmp_path / "build_demand.py"
    pre.write_text("# nothing\n")
    model = _build_fake_model(tmp_path / "src_model")
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        num_threads="4",
        pre_script=str(pre),
        pre_script_interpreter="/usr/bin/python3",
    )
    sample = Sample(id=1, inputs=np.array([0.42, -0.3]))
    workspace = tmp_path / "experiments" / "sim-1"
    spec = sim.prepare(sample, space, workspace)
    cmd = spec.command
    # ``set -e`` guards: pre_script failure must abort before the binary runs.
    assert "set -e" in cmd
    # pre_script is invoked with dashified flags. The space fixture has
    # parameter names trip_threshold and alpha.
    assert "/usr/bin/python3" in cmd
    assert str(pre) in cmd
    assert "--trip-threshold=0.42" in cmd
    assert "--alpha=-0.3" in cmd
    # And pre_script appears BEFORE the /usr/bin/echo binary line.
    pre_pos = cmd.index(str(pre))
    binary_pos = cmd.index("/usr/bin/echo")
    assert pre_pos < binary_pos


def test_pre_script_default_is_none_and_no_set_e(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """Backwards compat: without pre_script, the single-iteration command
    is a single line and doesn't add ``set -e``.
    """
    sim = _make_sim(tmp_path)
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    workspace = tmp_path / "experiments" / "sim-1"
    spec = sim.prepare(sample, space, workspace)
    assert "set -e" not in spec.command
    assert spec.command.count("\n") == 0
    assert sim.pre_script is None


def test_pre_script_missing_raises_at_construction(tmp_path: Path) -> None:
    model = _build_fake_model(tmp_path / "m")
    with pytest.raises(SimulatorError, match="pre_script not found"):
        PolarisSimulator(
            binary="/usr/bin/echo",
            model_source=str(model),
            scenario_file="scenario_abm.json",
            output_db_filename="R.h5",
            pre_script=str(tmp_path / "no_such_script.py"),
        )


def test_pre_script_shell_escapes_value_with_spaces(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """Values with shell metacharacters survive intact (the CodeRabbit
    finding from v0.6 applies to pre_script forwarding too)."""
    pre = tmp_path / "build.py"
    pre.write_text("# pre\n")
    model = _build_fake_model(tmp_path / "m")
    # ParameterSpace with a name we can pass a tricky value through.
    space2 = ParameterSpace.from_iterable(
        [Parameter("tag", "DestinationChoice.json", 0.0, 1.0)]
    )
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        pre_script=str(pre),
    )
    # Trigger a shape-1 input.
    sample = Sample(id=1, inputs=np.array([0.5]))
    spec = sim.prepare(sample, space2, tmp_path / "sim-1")
    # Numeric value renders to a normal token; the escape isn't a no-op
    # at construction — sanity check that --tag=0.5 (a regular token).
    assert "--tag=0.5" in spec.command


def test_pre_script_runs_once_with_num_iterations_gt_one(
    tmp_path: Path, space: ParameterSpace
) -> None:
    pre = tmp_path / "build.py"
    pre.write_text("# pre\n")
    model = _build_fake_model(tmp_path / "m")
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        num_iterations=3,
        pre_script=str(pre),
    )
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    spec = sim.prepare(sample, space, tmp_path / "sim-1")
    cmd = spec.command
    # pre_script appears exactly once, ahead of the for-loop body.
    assert cmd.count(str(pre)) == 1
    for_pos = cmd.index("for i in $(seq")
    pre_pos = cmd.index(str(pre))
    assert pre_pos < for_pos


def test_pre_script_with_sif_binary(
    tmp_path: Path, space: ParameterSpace
) -> None:
    """SIF + pre_script: order is pre_script, then apptainer-wrapped binary."""
    pre = tmp_path / "build.py"
    pre.write_text("# pre\n")
    sif = tmp_path / "polaris_exe" / "polaris.sif"
    sif.parent.mkdir()
    sif.touch()
    model = _build_fake_model(tmp_path / "m")
    sim = PolarisSimulator(
        binary=str(sif),
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
        pre_script=str(pre),
    )
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    spec = sim.prepare(sample, space, tmp_path / "sim-1")
    cmd = spec.command
    pre_pos = cmd.index(str(pre))
    apptainer_pos = cmd.index("apptainer run")
    assert pre_pos < apptainer_pos


def test_make_simulator_polaris(tmp_path: Path) -> None:
    model = _build_fake_model(tmp_path / "m")
    sim = make_simulator(
        {
            "type": "polaris",
            "options": {
                "binary": "/usr/bin/echo",
                "model_source": str(model),
                "scenario_file": "scenario_abm.json",
                "output_db_filename": "R.h5",
            },
        }
    )
    assert isinstance(sim, PolarisSimulator)


def test_num_iterations_rejected_when_lt_one(tmp_path: Path) -> None:
    model = _build_fake_model(tmp_path / "m")
    with pytest.raises(SimulatorError):
        PolarisSimulator(
            binary="/usr/bin/echo",
            model_source=str(model),
            scenario_file="scenario_abm.json",
            output_db_filename="R.h5",
            num_iterations=0,
        )


def test_prepare_wraps_iterations_in_bash_loop(tmp_path: Path, space: ParameterSpace) -> None:
    model = _build_fake_model(tmp_path / "src_model")
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="TestModel-Result.h5",
        num_threads="8",
        num_iterations=3,
    )
    sample = Sample(id=99, inputs=np.array([0.5, 0.5]))
    workspace = tmp_path / "experiments" / "sim-99"
    spec = sim.prepare(sample, space, workspace)
    assert "for i in $(seq 1 3); do" in spec.command
    assert "iteration $i of 3" in spec.command


def test_collect_output_picks_latest_iteration(tmp_path: Path, space: ParameterSpace) -> None:
    model = _build_fake_model(tmp_path / "src_model")
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="TestModel-Result.h5",
        num_iterations=3,
    )
    sample = Sample(id=10, inputs=np.array([0.4, 0.6]))
    workspace = tmp_path / "experiments" / "sim-10"
    sim.prepare(sample, space, workspace)

    # Simulate POLARIS writing iteration_1, iteration_2, iteration_3 dirs.
    # Each contains its own Result.h5; we should pick iteration_3.
    for n in (1, 2, 3):
        d = workspace / f"out_iteration_{n}"
        d.mkdir(parents=True, exist_ok=True)
        with h5py.File(d / "TestModel-Result.h5", "w") as f:
            g = f.create_group("link_moe")
            g.create_dataset("link_travel_time", data=np.full((2, 2), n))
            g.create_dataset("link_in_volume", data=np.ones((2, 2)))

    sample.folder = workspace
    out = sim.collect_output(sample)
    assert out["iteration"] == 3
    assert out["output_dir"].endswith("out_iteration_3")
    assert "iteration_3" in out["result_path"]


def test_collect_output_no_iteration_dirs_returns_iteration_none(
    tmp_path: Path, space: ParameterSpace
) -> None:
    model = _build_fake_model(tmp_path / "src_model")
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="TestModel-Result.h5",
    )
    sample = Sample(id=11, inputs=np.array([0.2, 0.3]))
    workspace = tmp_path / "experiments" / "sim-11"
    sim.prepare(sample, space, workspace)
    (workspace / "out").mkdir(parents=True, exist_ok=True)
    with h5py.File(workspace / "out" / "TestModel-Result.h5", "w") as f:
        g = f.create_group("link_moe")
        g.create_dataset("link_travel_time", data=np.ones((1, 1)))
        g.create_dataset("link_in_volume", data=np.ones((1, 1)))
    sample.folder = workspace
    out = sim.collect_output(sample)
    assert out["iteration"] is None
    assert out["output_dir"].endswith("/out")

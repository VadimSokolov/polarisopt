"""End-to-end sequential Bayesian-optimization on Branin via the master loop."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

torch = pytest.importorskip("torch")
from polarisopt.config import load_study_config
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.runner import StudyRunner
from polarisopt.utils.paths import workspace_layout


def _sequential_yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: branin-bo-{workspace.name}
        workspace: {workspace}
        seed: 11
        simulator:
          type: mock
          options:
            function: branin
        runner:
          type: local
          options: {{}}
        parameters:
          inline:
            - {{ name: x1, file: a.json, min: -5.0, max: 10.0 }}
            - {{ name: x2, file: a.json, min: 0.0, max: 15.0 }}
        metric:
          type: identity
          options:
            keys: value
        phases:
          - name: bo
            type: sequential
            warm_up:
              type: lhs
              options:
                n: 6
            generator:
              type: acquisition
              options:
                surrogate: {{ type: gp, options: {{}} }}
                acquisition:
                  type: qei
                  options:
                    mc_samples: 32
                    num_restarts: 2
                    raw_samples: 32
            batch_size: 2
            stop:
              type: max_iter
              options:
                n: 3
        """
    )


def test_sequential_bo_branin_improves(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    config_path = tmp_path / "study.yaml"
    # Beef up the budget so the assertion is robust against seed variance.
    text = _sequential_yaml(workspace).replace("n: 6", "n: 8").replace("n: 3", "n: 5")
    config_path.write_text(text)

    config = load_study_config(config_path)
    samples = StudyRunner(config).run()

    # 8 warm-up + 5 iterations * 2 q = 18 samples
    assert len(samples) == 18
    finished = [s for s in samples if s.status is SampleStatus.FINISHED]
    assert len(finished) >= 16  # tolerate the rare optimizer hiccup

    # iteration 0 = warm-up; iteration > 0 = BO loop
    warm = [float(s.metric[0]) for s in finished if s.iteration == 0]
    later = [float(s.metric[0]) for s in finished if s.iteration > 0]
    assert warm and later

    # BO-driven samples should reach a value at least as low as the warm-up baseline.
    assert min(later) <= min(warm), (
        f"BO did not improve over warm-up: warm={min(warm):.3f}, bo={min(later):.3f}"
    )


def test_sequential_restart_via_phase_state(tmp_path: Path) -> None:
    """Run a phase with max_iter=2, then bump to max_iter=4 and resume.
    The first run leaves a phase_state checkpoint; the second run should
    restore RNG and pick up the iteration counter."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)

    text1 = _sequential_yaml(workspace).replace("n: 3", "n: 1")
    text2 = _sequential_yaml(workspace).replace("n: 3", "n: 2")

    cfg1 = tmp_path / "s1.yaml"
    cfg1.write_text(text1)
    cfg2 = tmp_path / "s2.yaml"
    cfg2.write_text(text2)

    config1 = load_study_config(cfg1)
    StudyRunner(config1).run()

    layout = workspace_layout(workspace)
    store_a = SampleStore.open(layout["db"], config1.name)
    state_a = store_a.load_phase_state("bo")
    assert state_a is not None and state_a["iteration"] == 1
    n_before = store_a.count(phase="bo")

    # Now run config2 (max_iter=2). Should resume from checkpointed iteration=1 and do
    # exactly one more iteration before stopping.
    config2 = load_study_config(cfg2)
    StudyRunner(config2).run()
    store_b = SampleStore.open(layout["db"], config2.name)
    state_b = store_b.load_phase_state("bo")
    assert state_b is not None and state_b["iteration"] == 2
    n_after = store_b.count(phase="bo")
    assert n_after > n_before

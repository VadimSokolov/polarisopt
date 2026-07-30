"""Unit tests for v0.29 identifiability pre-flight (P5)."""

from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.config import load_study_config
from polarisopt.metrics import MomentSetMetric
from polarisopt.parameters import GaussianPrior, Parameter, ParameterSpace
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.identifiability import (
    IdentifiabilityError,
    IdentifiabilityReport,
    ParameterIdentifiability,
    format_identifiability,
    identifiability_as_dict,
    run_identifiability_analysis,
)
from polarisopt.utils.paths import workspace_layout


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def _write_demand_db(path: Path, rows: list[tuple[str, str, int]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE Trip (purpose TEXT, mode TEXT, n INTEGER)")
    conn.executemany("INSERT INTO Trip VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _seed_synthetic_store(
    workspace: Path, name: str, target_csv: Path, *, n: int = 40,
) -> SampleStore:
    """Synthetic (X, moment_vector) where only x0 drives the residuals.

    The residual vector has one element per row in ``target_csv``
    (matching what ``moment_set`` would produce), and every element is
    driven purely by x0: ``0.5 * x[0] + noise``. x[1] and x[2] never
    affect the metric — Sobol should therefore flag them as un-identified.
    """
    # Count target CSV rows (excluding the header) so the synthetic
    # metric vector width matches what the moment_set metric will
    # expect. Otherwise identifiability's width-consistency check
    # rejects the samples.
    n_moments = sum(1 for _ in target_csv.open()) - 1
    if n_moments < 2:
        raise RuntimeError(
            f"_seed_synthetic_store needs a target CSV with >=2 rows; "
            f"{target_csv} has {n_moments}"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], name)
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = rng.uniform(size=3)
        residuals = np.full(n_moments, 0.5 * x[0]) + 0.01 * rng.standard_normal(n_moments)
        s = store.add(Sample(phase="warmup", inputs=x))
        s.status = SampleStatus.FINISHED
        s.metric = residuals
        store.update(s)
    return store


def _make_moment_metric(target_csv: Path) -> MomentSetMetric:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target_csv),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
        }])


def _space_3d() -> ParameterSpace:
    return ParameterSpace.from_iterable([
        Parameter("x0", "a.json", 0.0, 1.0,
                  prior=GaussianPrior(mean=0.5, std=0.1)),
        Parameter("x1", "a.json", 0.0, 1.0,
                  prior=GaussianPrior(mean=0.5, std=0.1)),
        Parameter("x2", "a.json", 0.0, 1.0),  # no prior
    ])


def test_run_identifiability_classifies_active_and_inactive(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv,
        "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"],
    )
    store = _seed_synthetic_store(tmp_path / "ws", "id-test", target_csv, n=40)
    metric = _make_moment_metric(target_csv)
    report = run_identifiability_analysis(
        store, _space_3d(), metric=metric, n_sobol=256, threshold=0.05,
    )
    by_name = {p.name: p for p in report.parameters}
    assert by_name["x0"].is_identified, by_name["x0"]
    assert not by_name["x1"].is_identified, by_name["x1"]
    assert not by_name["x2"].is_identified, by_name["x2"]


def test_run_identifiability_suggests_pin_only_when_prior_set(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv, "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"],
    )
    store = _seed_synthetic_store(tmp_path / "ws", "id-pins", target_csv, n=40)
    report = run_identifiability_analysis(
        store, _space_3d(), metric=_make_moment_metric(target_csv), n_sobol=256,
    )
    by_name = {p.name: p for p in report.parameters}
    # x1 unidentified AND prior=0.5 → suggested_pin=0.5
    assert by_name["x1"].suggested_pin == pytest.approx(0.5)
    # x2 unidentified but no prior → no pin
    assert by_name["x2"].suggested_pin is None


def test_run_identifiability_records_per_moment_indices(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv, "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"],
    )
    store = _seed_synthetic_store(tmp_path / "ws", "id-perm", target_csv, n=40)
    report = run_identifiability_analysis(
        store, _space_3d(), metric=_make_moment_metric(target_csv), n_sobol=256,
    )
    x0 = next(p for p in report.parameters if p.name == "x0")
    # 'shares' is the only moment; every parameter's per_moment_first_order
    # dict has exactly that key.
    assert set(x0.per_moment_first_order) == {"shares"}
    assert set(x0.per_moment_total) == {"shares"}


def test_run_identifiability_rejects_scalar_metric(tmp_path: Path) -> None:
    """Scalar metrics belong to `polarisopt sensitivity`, not this CLI."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "scalar")
    for i in range(5):
        s = store.add(Sample(phase="w", inputs=np.array([0.1 * i, 0.5, 0.3])))
        s.status = SampleStatus.FINISHED
        s.metric = np.array([0.42])  # scalar
        store.update(s)
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    with pytest.raises(IdentifiabilityError, match="multi-moment"):
        run_identifiability_analysis(
            store, _space_3d(), metric=_make_moment_metric(target_csv),
            n_sobol=64,
        )


def test_run_identifiability_rejects_inconsistent_widths(tmp_path: Path) -> None:
    """Stored samples with different metric widths surface a clear error,
    not a numpy shape crash inside the GP fit."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "widthmix")
    for i in range(5):
        s = store.add(Sample(phase="w", inputs=np.array([0.1 * i, 0.5, 0.3])))
        s.status = SampleStatus.FINISHED
        s.metric = np.zeros(3 if i % 2 == 0 else 2)
        store.update(s)
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    with pytest.raises(IdentifiabilityError, match="inconsistent metric widths"):
        run_identifiability_analysis(
            store, _space_3d(), metric=_make_moment_metric(target_csv), n_sobol=64,
        )


def test_run_identifiability_rejects_empty_store(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "empty")
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    with pytest.raises(IdentifiabilityError, match="no FINISHED"):
        run_identifiability_analysis(
            store, _space_3d(), metric=_make_moment_metric(target_csv), n_sobol=64,
        )


def test_format_identifiability_output_shape() -> None:
    report = IdentifiabilityReport(
        parameters=[
            ParameterIdentifiability(
                name="a", max_first_order=0.7, max_total=0.85,
                is_identified=True, suggested_pin=None,
                per_moment_first_order={"m": 0.7}, per_moment_total={"m": 0.85},
            ),
            ParameterIdentifiability(
                name="b", max_first_order=0.01, max_total=0.02,
                is_identified=False, suggested_pin=0.5,
                per_moment_first_order={"m": 0.01}, per_moment_total={"m": 0.02},
            ),
        ],
        moment_names=("m",),
        n_train=40, n_sobol=256, threshold=0.05,
    )
    text = format_identifiability(report)
    # Identified param appears first (sorted by max_S1 desc).
    assert text.index("a") < text.index("b")
    assert "IDENTIFIED" in text
    assert "unidentified" in text
    assert "1 / 2" in text  # Identified count


def test_identifiability_as_dict_is_json_roundtrippable() -> None:
    report = IdentifiabilityReport(
        parameters=[
            ParameterIdentifiability(
                name="a", max_first_order=0.7, max_total=0.85,
                is_identified=True, suggested_pin=None,
                per_moment_first_order={"m": 0.7}, per_moment_total={"m": 0.85},
            ),
        ],
        moment_names=("m",), n_train=40, n_sobol=256, threshold=0.05,
    )
    payload = identifiability_as_dict(report)
    round = json.loads(json.dumps(payload))
    assert round["parameters"][0]["name"] == "a"
    assert round["parameters"][0]["is_identified"] is True
    assert round["threshold"] == 0.05


def _write_id_yaml(workspace: Path, target_csv: Path) -> str:
    return dedent(
        f"""\
        name: id-{workspace.name}
        workspace: {workspace}
        simulator:
          type: mock
          options: {{ function: quadratic }}
        runner:
          type: local
          options: {{}}
        parameters:
          inline:
            - {{ name: x0, file: a.json, min: 0.0, max: 1.0, prior: {{ type: gaussian, mean: 0.5, std: 0.1 }} }}
            - {{ name: x1, file: a.json, min: 0.0, max: 1.0, prior: {{ type: gaussian, mean: 0.5, std: 0.1 }}, hold_at_prior_mean_if_unidentified: true }}
            - {{ name: x2, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: moment_set
          options:
            source_key: demand_db
            scalarize: none
            moments:
              - name: shares
                source_sql: "SELECT purpose, mode, 1.0 AS share FROM Trip"
                target: {target_csv}
                target_key_cols: [purpose, mode]
                target_value_col: share
                model_discrepancy_std: 0.01
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5, 0.5, 0.5]]
        """
    )


def test_cli_identifiability_prints_table(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv, "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"],
    )
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_write_id_yaml(workspace, target_csv))
    cfg = load_study_config(cfg_path)
    _seed_synthetic_store(workspace, cfg.name, target_csv, n=40)
    res = CliRunner().invoke(cli, ["identifiability", str(cfg_path), "--n-sobol", "256"])
    assert res.exit_code == 0, res.output
    assert "Identifiability report" in res.output
    assert "x0" in res.output
    assert "IDENTIFIED" in res.output


def test_cli_identifiability_json(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_write_id_yaml(workspace, target_csv))
    cfg = load_study_config(cfg_path)
    _seed_synthetic_store(workspace, cfg.name, target_csv, n=40)
    res = CliRunner().invoke(
        cli, ["identifiability", str(cfg_path), "--n-sobol", "256", "--json"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert "parameters" in payload
    names = [p["name"] for p in payload["parameters"]]
    assert set(names) == {"x0", "x1", "x2"}


def test_cli_identifiability_auto_drop_rewrites_yaml(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_write_id_yaml(workspace, target_csv))
    cfg = load_study_config(cfg_path)
    _seed_synthetic_store(workspace, cfg.name, target_csv, n=40)
    res = CliRunner().invoke(
        cli, ["identifiability", str(cfg_path), "--n-sobol", "256",
              "--auto-drop-to-prior-mean"],
    )
    assert res.exit_code == 0, res.output
    # x1 is unidentified AND has hold_at_prior_mean_if_unidentified=true,
    # so it should be pinned. x2 is unidentified but no flag, so skipped.
    assert "Pinned 1 parameter" in res.output
    assert "x1" in res.output.split("Pinned")[1]
    # New file with .pinned suffix exists
    pinned_path = cfg_path.with_suffix(".pinned.yaml")
    assert pinned_path.exists()
    import yaml
    with pinned_path.open() as fh:
        new_cfg = yaml.safe_load(fh)
    x1 = next(r for r in new_cfg["parameters"]["inline"] if r["name"] == "x1")
    # min ≈ max ≈ 0.5 (prior mean); tiny box around the pin. Halfwidth
    # is max(|pin|·1e-3, 1e-6) so |min − max| = 2·halfwidth ≈ 1e-3 here.
    assert x1["min"] == pytest.approx(0.5, abs=1e-3)
    assert x1["max"] == pytest.approx(0.5, abs=1e-3)
    assert x1["max"] > x1["min"]  # non-degenerate box, LHS won't crash
    assert x1["_pinned_at"] == pytest.approx(0.5)


def test_cli_identifiability_rejects_scalar_metric(tmp_path: Path) -> None:
    """A study whose metric is choice_share (scalar) surfaces a clean
    ClickException, not a stack trace."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (tmp_path / "target.sqlite").write_bytes(b"")  # avoid load-time failure
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(dedent(f"""
        name: scalar-{workspace.name}
        workspace: {workspace}
        simulator:
          type: mock
          options: {{ function: quadratic }}
        runner:
          type: local
          options: {{}}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: identity
          options: {{ keys: value }}
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
    """))
    res = CliRunner().invoke(cli, ["identifiability", str(cfg_path)])
    assert res.exit_code != 0
    assert "moment_set metric" in res.output

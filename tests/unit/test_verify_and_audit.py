"""Unit tests for v0.30 CLI subcommands verify-metric and discrepancy-audit,
and for the polarisopt.moments helper module."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.moments import (
    boarding_by_agency,
    mean_travel_time_by_activity,
    mode_shares_by_purpose,
    trip_distance_deciles_by_mode,
)


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def _write_demand_db(path: Path, rows: list[tuple[str, str, int]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE Trip (purpose TEXT, mode TEXT, n INTEGER)")
    conn.executemany("INSERT INTO Trip VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


# ----- polarisopt.moments helper module (P6) -----


def test_mode_shares_by_purpose_shape() -> None:
    m = mode_shares_by_purpose(target="/tmp/targets.csv")
    assert m["name"] == "mode_shares_by_purpose"
    assert m["target_key_cols"] == ["purpose", "mode"]
    assert m["target_value_col"] == "share"
    assert "GROUP BY purpose, mode" in m["source_sql"]
    assert m["aggregation"] == "elementwise_residual"
    assert m["model_discrepancy_std"] == 0.02


def test_boarding_by_agency_defaults_log_ratio() -> None:
    """Boarding counts span decades; log_ratio_residual is the sane default."""
    m = boarding_by_agency(target="/tmp/b.csv")
    assert m["aggregation"] == "log_ratio_residual"
    assert m["target_key_cols"] == ["agency", "type"]


def test_mean_travel_time_by_activity_units_are_minutes() -> None:
    """DFW defaults: obs=2 min, md=4 min."""
    m = mean_travel_time_by_activity(target="/tmp/tt.csv")
    assert m["obs_noise_std"] == 2.0
    assert m["model_discrepancy_std"] == 4.0


def test_trip_distance_deciles_by_mode_uses_ntile() -> None:
    m = trip_distance_deciles_by_mode(target="/tmp/d.csv")
    assert "NTILE(10)" in m["source_sql"]
    assert m["target_key_cols"] == ["mode", "decile"]


def test_all_helpers_accept_override_kwargs() -> None:
    """Every helper takes obs_noise_std / model_discrepancy_std / weight /
    aggregation / name overrides — required for user customization."""
    for fn in (
        mode_shares_by_purpose,
        boarding_by_agency,
        mean_travel_time_by_activity,
        trip_distance_deciles_by_mode,
    ):
        m = fn(
            target="/tmp/x.csv",
            obs_noise_std=99.9,
            model_discrepancy_std=88.8,
            weight_per_element=7.0,
            name="my-moment",
        )
        assert m["obs_noise_std"] == 99.9
        assert m["model_discrepancy_std"] == 88.8
        assert m["weight_per_element"] == 7.0
        assert m["name"] == "my-moment"


# ----- verify-metric CLI (P7) -----


def _yaml_choice_share(workspace: Path, target_db: Path, aggregation: str) -> str:
    return dedent(
        f"""\
        name: verify-{workspace.name}
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
          type: choice_share
          options:
            target_db: {target_db}
            sql: "SELECT mode AS category, n AS count FROM Trip"
            aggregation: {aggregation}
            laplace_smoothing_alpha: 0   # so identity assertions hold exactly
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
        """
    )


def test_verify_metric_choice_share_kl_zero(tmp_path: Path) -> None:
    """KL(target||target) = 0 exactly with alpha=0."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    target_db = tmp_path / "target.sqlite"
    conn = sqlite3.connect(str(target_db))
    conn.execute("CREATE TABLE Trip (mode TEXT, n INTEGER)")
    conn.executemany("INSERT INTO Trip VALUES (?, ?)",
                     [("auto", 60), ("transit", 40)])
    conn.commit()
    conn.close()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml_choice_share(workspace, target_db, "kl_divergence"))
    res = CliRunner().invoke(
        cli, ["verify-metric", str(cfg_path), "--reference-db", str(target_db)],
    )
    assert res.exit_code == 0, res.output
    assert "OK — kl_divergence" in res.output


def test_verify_metric_choice_share_js_zero(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    target_db = tmp_path / "target.sqlite"
    conn = sqlite3.connect(str(target_db))
    conn.execute("CREATE TABLE Trip (mode TEXT, n INTEGER)")
    conn.executemany("INSERT INTO Trip VALUES (?, ?)", [("auto", 60), ("transit", 40)])
    conn.commit()
    conn.close()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml_choice_share(workspace, target_db, "jensen_shannon"))
    res = CliRunner().invoke(
        cli, ["verify-metric", str(cfg_path), "--reference-db", str(target_db)],
    )
    assert res.exit_code == 0, res.output
    assert "OK — jensen_shannon" in res.output


def test_verify_metric_cross_entropy_reports_entropy(tmp_path: Path) -> None:
    """CE has no zero-identity — polarisopt just prints the value so the
    user can compare to their target's H(p)."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    target_db = tmp_path / "target.sqlite"
    conn = sqlite3.connect(str(target_db))
    conn.execute("CREATE TABLE Trip (mode TEXT, n INTEGER)")
    conn.executemany("INSERT INTO Trip VALUES (?, ?)", [("auto", 60), ("transit", 40)])
    conn.commit()
    conn.close()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml_choice_share(workspace, target_db, "cross_entropy"))
    res = CliRunner().invoke(
        cli, ["verify-metric", str(cfg_path), "--reference-db", str(target_db)],
    )
    assert res.exit_code == 0, res.output
    assert "cross_entropy = " in res.output


def test_verify_metric_moment_set_zero_residuals(tmp_path: Path) -> None:
    """Identity target → every moment_set residual ≈ 0."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.6", "HBW,transit,0.4"])
    ref_db = tmp_path / "ref.sqlite"
    _write_demand_db(ref_db, [("HBW", "auto", 60), ("HBW", "transit", 40)])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(dedent(f"""
        name: mv-{workspace.name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: moment_set
          options:
            source_key: demand_db
            scalarize: none
            moments:
              - name: shares
                source_sql: |
                  SELECT purpose, mode,
                         COUNT(*) * 1.0 / (SELECT COUNT(*) FROM Trip WHERE purpose IS NOT NULL) AS share
                  FROM Trip WHERE purpose IS NOT NULL
                  GROUP BY purpose, mode
                target: {target_csv}
                target_key_cols: [purpose, mode]
                target_value_col: share
                obs_noise_std: 0.005
                model_discrepancy_std: 0.02
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
    """))
    res = CliRunner().invoke(
        cli, ["verify-metric", str(cfg_path), "--reference-db", str(ref_db)],
    )
    assert res.exit_code == 0, res.output
    assert "OK — every moment residual within tolerance" in res.output


def test_verify_metric_moment_set_mismatched_target_flags_nonzero(tmp_path: Path) -> None:
    """A target CSV whose values disagree with the reference DB surfaces
    a nonzero residual and exits non-zero."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    target_csv = tmp_path / "t.csv"
    # Target says 50-50, ref DB is 60-40 → residual 0.1
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    ref_db = tmp_path / "ref.sqlite"
    _write_demand_db(ref_db, [("HBW", "auto", 60), ("HBW", "transit", 40)])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(dedent(f"""
        name: mv-fail-{workspace.name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: moment_set
          options:
            source_key: demand_db
            scalarize: none
            moments:
              - name: shares
                source_sql: |
                  SELECT purpose, mode,
                         COUNT(*) * 1.0 / (SELECT COUNT(*) FROM Trip WHERE purpose IS NOT NULL) AS share
                  FROM Trip WHERE purpose IS NOT NULL
                  GROUP BY purpose, mode
                target: {target_csv}
                target_key_cols: [purpose, mode]
                target_value_col: share
                obs_noise_std: 0.005
                model_discrepancy_std: 0.02
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
    """))
    res = CliRunner().invoke(
        cli, ["verify-metric", str(cfg_path), "--reference-db", str(ref_db),
              "--tolerance", "1e-6"],
    )
    assert res.exit_code != 0
    assert "exceed tolerance" in res.output


# ----- discrepancy-audit CLI (P10 full) -----


def _yaml_with_moment_set(workspace: Path, target_csv: Path, md: float) -> str:
    return dedent(f"""
        name: audit-{workspace.name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
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
                obs_noise_std: 0.005
                model_discrepancy_std: {md}
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
    """)


def test_discrepancy_audit_pass(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml_with_moment_set(workspace, target_csv, md=0.02))
    res = CliRunner().invoke(cli, ["discrepancy-audit", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "OK" in res.output


def test_discrepancy_audit_fails_on_zero_md(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml_with_moment_set(workspace, target_csv, md=0.0))
    res = CliRunner().invoke(cli, ["discrepancy-audit", str(cfg_path)])
    assert res.exit_code != 0
    assert "FAIL" in res.output
    assert "Vernon" in res.output


def test_discrepancy_audit_rejects_scalar_metric(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(dedent(f"""
        name: audit-scalar-{workspace.name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        runner: {{ type: local, options: {{}} }}
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
    res = CliRunner().invoke(cli, ["discrepancy-audit", str(cfg_path)])
    assert res.exit_code != 0
    assert "moment_set metric" in res.output

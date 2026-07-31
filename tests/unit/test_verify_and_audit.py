"""Unit tests for v0.30 CLI subcommands verify-metric and discrepancy-audit,
and for the polarisopt.moments helper module."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.moments import (
    mean_travel_time_by_activity,
    mode_shares_by_purpose,
    trip_distance_deciles_by_mode,
    trip_mode_shares,
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
#
# These tests EXECUTE the helper SQL against a SQLite database built to
# the real polarislib Demand schema. The v0.30 versions only asserted
# the returned dict's *shape*, which is why four helpers shipped with
# SQL that referenced nonexistent columns (Activity.activity_id,
# Trip.travel_time, MM_Trip.agency) and wrong units. Shape-only tests
# cannot catch that; execution tests can.


def _make_demand_db(path: Path) -> None:
    """Build a Demand SQLite matching the polarislib schema for the
    columns these helpers touch.

    Mirrors polaris/demand/database/sql_schema/{activity,trip,mode}.sql:
      Activity.mode is TEXT, Activity.type is TEXT (purpose),
      Activity.trip is 0 for planned activities.
      Trip.mode is INTEGER (FK -> Mode.mode_id), travel_distance is
      METERS, start/end are SECONDS, there is no travel_time column.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE "Activity" (
          "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
          "seq_num" INTEGER NOT NULL DEFAULT 0,
          "location_id" INTEGER NOT NULL DEFAULT 0,
          "start_time" REAL NULL DEFAULT 0,
          "duration" REAL NULL DEFAULT 0,
          "mode" TEXT NOT NULL DEFAULT '',
          "type" TEXT NOT NULL DEFAULT '',
          "person" INTEGER NOT NULL,
          "trip" INTEGER NOT NULL,
          "origin_id" INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE "Trip" (
          "trip_id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
          "origin" INTEGER NOT NULL DEFAULT 0,
          "destination" INTEGER NOT NULL DEFAULT 0,
          "purpose" INTEGER NOT NULL DEFAULT 0,
          "mode" INTEGER NOT NULL DEFAULT 0,
          "start" REAL NULL DEFAULT 0,
          "end" REAL NULL DEFAULT 0,
          "travel_distance" REAL NULL DEFAULT 0,
          "person" INTEGER NULL);
        CREATE TABLE "Mode" (
          "mode_id" INTEGER NOT NULL PRIMARY KEY,
          "mode_description" TEXT NOT NULL);
        """
    )
    conn.executemany(
        "INSERT INTO Mode VALUES (?, ?)",
        [(0, "SOV"), (1, "HOV"), (2, "BUS"), (3, "WALK")],
    )
    # 4 trips: 2 SOV, 1 BUS, 1 WALK. Distances in METERS.
    # travel time = end - start, in SECONDS (600s = 10 min).
    conn.executemany(
        "INSERT INTO Trip (trip_id, mode, start, end, travel_distance) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 0, 0.0, 600.0, 16093.44),    # SOV, 10 min, 10 mi
            (2, 0, 0.0, 1200.0, 32186.88),   # SOV, 20 min, 20 mi
            (3, 2, 0.0, 1800.0, 8046.72),    # BUS, 30 min, 5 mi
            (4, 3, 0.0, 300.0, 1609.344),    # WALK, 5 min, 1 mi
        ],
    )
    # Activities: 4 executed (linked to trips) + 1 planned (trip=0)
    # + 1 with no mode (must be excluded from shares entirely).
    conn.executemany(
        "INSERT INTO Activity (mode, type, person, trip) VALUES (?, ?, ?, ?)",
        [
            ("SOV", "HBW", 1, 1),
            ("SOV", "HBW", 2, 2),
            ("BUS", "HBO", 3, 3),
            ("WALK", "HBO", 4, 4),
            ("SOV", "HBW", 5, 0),        # planned only
            ("", "HBO", 6, 0),           # no mode -> excluded
            ("NO_MOVE", "HBO", 7, 0),    # NO_MOVE -> excluded
        ],
    )
    conn.commit()
    conn.close()


def _run_moment_sql(db: Path, moment: dict) -> list[tuple]:
    """Execute a helper's source_sql and return rows. Verifies the SQL
    is valid against the schema and returns the declared columns."""
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute(moment["source_sql"])
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
    finally:
        conn.close()
    for declared in (*moment["target_key_cols"], moment["target_value_col"]):
        assert declared in cols, (
            f"{moment['name']}: SQL does not return declared column "
            f"{declared!r}; got {cols}"
        )
    return rows


def test_mode_shares_by_purpose_sql_executes_and_sums_to_one(tmp_path: Path) -> None:
    """Activity-based shares: 5 mode-carrying activities (SOV x3, BUS,
    WALK) -> shares sum to 1; '' and NO_MOVE excluded."""
    db = tmp_path / "demand.sqlite"
    _make_demand_db(db)
    m = mode_shares_by_purpose(target="/tmp/x.csv")
    rows = _run_moment_sql(db, m)
    shares = {(r[0], r[1]): r[2] for r in rows}
    # SOV/HBW = 3 of 5 mode-carrying activities (2 executed + 1 planned).
    assert shares[("HBW", "SOV")] == pytest.approx(3 / 5)
    assert shares[("HBO", "BUS")] == pytest.approx(1 / 5)
    assert shares[("HBO", "WALK")] == pytest.approx(1 / 5)
    assert sum(shares.values()) == pytest.approx(1.0)
    # Excluded non-modes never appear as keys.
    assert not any(mode in ("", "NO_MOVE") for _, mode in shares)


def test_mode_shares_by_purpose_planned_only_filter(tmp_path: Path) -> None:
    """planned_only=True restricts to trip=0 — the pure choice-model
    output a nested-ASC contraction produces."""
    db = tmp_path / "demand.sqlite"
    _make_demand_db(db)
    m = mode_shares_by_purpose(target="/tmp/x.csv", planned_only=True)
    rows = _run_moment_sql(db, m)
    shares = {(r[0], r[1]): r[2] for r in rows}
    # Only one planned activity carries a mode: SOV/HBW.
    assert shares == {("HBW", "SOV"): pytest.approx(1.0)}


def test_trip_mode_shares_sql_joins_mode_table(tmp_path: Path) -> None:
    """Trip.mode is an INTEGER key; the helper must join Mode to yield
    readable text keys matching a target CSV."""
    db = tmp_path / "demand.sqlite"
    _make_demand_db(db)
    m = trip_mode_shares(target="/tmp/x.csv")
    rows = _run_moment_sql(db, m)
    shares = {r[0]: r[1] for r in rows}
    assert shares["SOV"] == pytest.approx(2 / 4)
    assert shares["BUS"] == pytest.approx(1 / 4)
    assert shares["WALK"] == pytest.approx(1 / 4)
    # Keys are text, not integers.
    assert all(isinstance(k, str) for k in shares)


def test_mean_travel_time_by_activity_returns_minutes(tmp_path: Path) -> None:
    """Trip stores start/end in SECONDS and has no travel_time column.
    The helper must compute (end-start)/60 so the value is in minutes,
    matching the documented obs_noise_std units."""
    db = tmp_path / "demand.sqlite"
    _make_demand_db(db)
    m = mean_travel_time_by_activity(target="/tmp/x.csv")
    rows = _run_moment_sql(db, m)
    tt = {r[0]: r[1] for r in rows}
    # HBW: trips 1 and 2 -> (10 + 20) / 2 = 15 min.
    assert tt["HBW"] == pytest.approx(15.0)
    # HBO: trips 3 and 4 -> (30 + 5) / 2 = 17.5 min.
    assert tt["HBO"] == pytest.approx(17.5)


def test_mean_travel_time_excludes_planned_activities(tmp_path: Path) -> None:
    """Planned activities (trip=0) have no executed trip; including them
    would inject a spurious 0-minute observation."""
    db = tmp_path / "demand.sqlite"
    _make_demand_db(db)
    m = mean_travel_time_by_activity(target="/tmp/x.csv")
    rows = _run_moment_sql(db, m)
    # If planned rows leaked in, HBW's mean would be pulled below 15.
    tt = {r[0]: r[1] for r in rows}
    assert tt["HBW"] == pytest.approx(15.0)


def test_trip_distance_deciles_returns_miles(tmp_path: Path) -> None:
    """travel_distance is METERS in the schema; the helper converts to
    miles so the value matches the documented obs_noise_std units."""
    db = tmp_path / "demand.sqlite"
    _make_demand_db(db)
    m = trip_distance_deciles_by_mode(target="/tmp/x.csv")
    rows = _run_moment_sql(db, m)
    by_mode = {}
    for mode, _decile, dist in rows:
        by_mode.setdefault(mode, []).append(dist)
    # WALK has a single 1609.344 m trip -> exactly 1.0 mile.
    assert max(by_mode["WALK"]) == pytest.approx(1.0)
    # SOV max is the 20-mile trip, not 32186.88.
    assert max(by_mode["SOV"]) == pytest.approx(20.0)


def test_all_helpers_execute_against_real_schema(tmp_path: Path) -> None:
    """Every shipped helper must produce SQL that runs and returns its
    declared key/value columns. This is the regression guard for the
    v0.30 class of bug (SQL referencing nonexistent columns)."""
    db = tmp_path / "demand.sqlite"
    _make_demand_db(db)
    for fn in (
        mode_shares_by_purpose,
        trip_mode_shares,
        mean_travel_time_by_activity,
        trip_distance_deciles_by_mode,
    ):
        moment = fn(target="/tmp/x.csv")
        rows = _run_moment_sql(db, moment)
        assert rows, f"{moment['name']} returned no rows on a populated DB"


def test_all_helpers_accept_override_kwargs() -> None:
    """Every helper takes obs_noise_std / model_discrepancy_std / weight /
    aggregation / name overrides — required for user customization."""
    for fn in (
        mode_shares_by_purpose,
        trip_mode_shares,
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


def test_boarding_by_agency_no_longer_exported() -> None:
    """v0.36 removed boarding_by_agency: it queried MM_Trip.agency, and
    no agency column exists anywhere in the Demand schema (MM_Trip is
    the micromobility table). Importing it must fail loudly rather than
    silently resolve to broken SQL."""
    import polarisopt.moments as moments_mod

    assert not hasattr(moments_mod, "boarding_by_agency")
    assert "boarding_by_agency" not in moments_mod.__all__



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
                         SUM(n) * 1.0 / (SELECT SUM(n) FROM Trip WHERE purpose IS NOT NULL) AS share
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
                         SUM(n) * 1.0 / (SELECT SUM(n) FROM Trip WHERE purpose IS NOT NULL) AS share
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

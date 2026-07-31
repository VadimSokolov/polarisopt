"""Reusable moment builders for :class:`polarisopt.metrics.MomentSetMetric`.

v0.30 P6; **SQL corrected in v0.36** against the real polarislib Demand
schema (``polaris/demand/database/sql_schema/``). Each helper returns a
canned ``moment`` dict that drops straight into a ``moment_set.moments``
YAML list, so users don't re-implement the same SQL every study.

Schema facts these helpers rely on (verified against polarislib):

- ``Activity(id, seq_num, location_id, start_time, duration, mode TEXT,
  type TEXT, person, trip, origin_id)`` — ``mode`` is the **text** mode
  key (``SOV``/``HOV``/``BUS``/``WALK``/…) and ``type`` is the activity
  purpose (``HBW``/``HBO``/…). ``trip = 0`` marks a *planned* activity
  with no executed trip.
- ``Trip(trip_id, …, origin, destination, purpose INTEGER, mode INTEGER,
  travel_distance REAL /* meters */, start REAL, end REAL /* seconds */,
  …)`` — ``Trip.mode`` is an **integer** key into ``Mode``, and
  ``Trip.purpose`` is *not* the activity purpose (schema comment:
  "currently used only to distinguish freight trips as E-Commerce or
  not"). Executed travel time is ``end - start``; there is no
  ``travel_time`` column.
- ``Mode(mode_id INTEGER, mode_description TEXT)`` — join target to turn
  ``Trip.mode`` into readable text.

**Activity vs Trip for mode-choice calibration.** ``Activity.mode`` is
the direct output of the ADAPTS mode-choice model; ``Trip.mode`` is the
executed mode after DTA reshuffling and reflects downstream layers
(routing, congestion). For β-calibration of mode choice specifically,
Activity is the cleaner signal — see the DFW v0.32 report. Helpers here
default to Activity where that choice applies, with Trip-based variants
provided explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Modes that mean "no movement" and should never enter a share
# denominator. Empty string is the schema default for a planned
# activity that was never assigned a mode.
_NON_MODES = ("''", "'NO_MOVE'")
_MODE_FILTER = f"mode NOT IN ({', '.join(_NON_MODES)})"


def mode_shares_by_purpose(
    *,
    target: Path | str,
    obs_noise_std: float = 0.005,
    model_discrepancy_std: float = 0.02,
    weight_per_element: float = 1.0,
    aggregation: str = "elementwise_residual",
    name: str = "mode_shares_by_purpose",
    planned_only: bool = False,
) -> dict[str, Any]:
    """POM moment #1: mode share by ``(purpose, mode)`` from ``Activity``.

    Target CSV columns: ``purpose``, ``mode``, ``share``. Shares are
    normalized over all mode-carrying activities, so the full table
    sums to 1 (not per-purpose).

    Uses ``Activity.type`` as the purpose and ``Activity.mode`` as the
    (text) mode — the direct mode-choice-model output. Activities with
    no mode (``''``) or ``NO_MOVE`` are excluded from both numerator
    and denominator.

    Parameters
    ----------
    planned_only : bool, optional
        When True, restrict to *planned* activities (``trip = 0``) —
        the pure choice-model output before DTA assignment. This is
        what a nested-ASC contraction run produces in its
        ``_calib_N`` directories. Default False (all activities).

    Defaults ``obs_noise_std=0.005`` / ``model_discrepancy_std=0.02``
    follow the DFW β-calibration design's Table 1.
    """
    where = [_MODE_FILTER]
    if planned_only:
        where.append("trip = 0")
    where_sql = " AND ".join(where)
    return {
        "name": name,
        "source_sql": (
            f"SELECT type AS purpose, mode, "  # noqa: S608 - static SQL, no user interpolation
            f"COUNT(*) * 1.0 / (SELECT COUNT(*) FROM Activity WHERE {where_sql}) AS share "
            f"FROM Activity WHERE {where_sql} "
            f"GROUP BY type, mode"
        ),
        "target": str(target),
        "target_key_cols": ["purpose", "mode"],
        "target_value_col": "share",
        "obs_noise_std": obs_noise_std,
        "model_discrepancy_std": model_discrepancy_std,
        "weight_per_element": weight_per_element,
        "aggregation": aggregation,
    }


def trip_mode_shares(
    *,
    target: Path | str,
    obs_noise_std: float = 0.005,
    model_discrepancy_std: float = 0.02,
    weight_per_element: float = 1.0,
    aggregation: str = "elementwise_residual",
    name: str = "trip_mode_shares",
) -> dict[str, Any]:
    """Executed (post-DTA) mode shares from ``Trip``, keyed by mode text.

    Target CSV columns: ``mode``, ``share``. Joins ``Trip.mode``
    (integer) to ``Mode.mode_description`` so the target CSV can use
    readable mode names.

    Use this when you care about the mode mix *after* routing and
    congestion feedback. For calibrating mode-choice β's directly,
    prefer :func:`mode_shares_by_purpose` (Activity-based).
    """
    return {
        "name": name,
        "source_sql": (
            "SELECT m.mode_description AS mode, "
            "COUNT(*) * 1.0 / (SELECT COUNT(*) FROM Trip) AS share "
            "FROM Trip t JOIN Mode m ON t.mode = m.mode_id "
            "GROUP BY m.mode_description"
        ),
        "target": str(target),
        "target_key_cols": ["mode"],
        "target_value_col": "share",
        "obs_noise_std": obs_noise_std,
        "model_discrepancy_std": model_discrepancy_std,
        "weight_per_element": weight_per_element,
        "aggregation": aggregation,
    }


def mean_travel_time_by_activity(
    *,
    target: Path | str,
    obs_noise_std: float = 2.0,
    model_discrepancy_std: float = 4.0,
    weight_per_element: float = 1.0,
    aggregation: str = "elementwise_residual",
    name: str = "mean_travel_time_by_activity",
) -> dict[str, Any]:
    """POM moment #3: mean executed travel time (**minutes**) by activity type.

    Target CSV columns: ``dest_act_type``, ``mean_travel_time``.

    Joins ``Activity.trip = Trip.trip_id`` (the correct linkage —
    ``Trip.destination`` is a *Location* foreign key, not an Activity
    id) and computes travel time as ``Trip.end - Trip.start``, which
    the schema stores in **seconds**; the SQL divides by 60 so the
    metric and its ``obs_noise_std`` are both in minutes.

    Planned activities (``trip = 0``) are excluded — they have no
    executed trip and would otherwise contribute a spurious 0.
    """
    return {
        "name": name,
        "source_sql": (
            "SELECT a.type AS dest_act_type, "
            "AVG((t.end - t.start) / 60.0) AS mean_travel_time "
            "FROM Activity a JOIN Trip t ON a.trip = t.trip_id "
            "WHERE a.trip <> 0 AND t.end IS NOT NULL AND t.start IS NOT NULL "
            "GROUP BY a.type"
        ),
        "target": str(target),
        "target_key_cols": ["dest_act_type"],
        "target_value_col": "mean_travel_time",
        "obs_noise_std": obs_noise_std,
        "model_discrepancy_std": model_discrepancy_std,
        "weight_per_element": weight_per_element,
        "aggregation": aggregation,
    }


def trip_distance_deciles_by_mode(
    *,
    target: Path | str,
    obs_noise_std: float = 0.5,
    model_discrepancy_std: float = 1.0,
    weight_per_element: float = 1.0,
    aggregation: str = "elementwise_residual",
    name: str = "trip_distance_deciles_by_mode",
) -> dict[str, Any]:
    """POM moment #4: trip-distance deciles (**miles**) by mode.

    Target CSV columns: ``mode``, ``decile`` (1..10), ``distance``.

    ``Trip.travel_distance`` is stored in **meters**; the SQL converts
    to miles (``/ 1609.344``) so the value and its ``obs_noise_std``
    (default 0.5 mi) share units. ``Trip.mode`` is joined to ``Mode``
    for readable keys.

    Uses ``NTILE(10) OVER (PARTITION BY mode ORDER BY travel_distance)``
    and takes the max per ``(mode, decile)`` as the decile boundary —
    a coarse but robust empirical-CDF approximation. Requires SQLite
    ≥ 3.25 for window functions (bundled with Python ≥ 3.9).
    """
    return {
        "name": name,
        "source_sql": (
            "SELECT mode, decile, MAX(distance_mi) AS distance FROM ("
            "  SELECT m.mode_description AS mode, "
            "         t.travel_distance / 1609.344 AS distance_mi, "
            "         NTILE(10) OVER ("
            "           PARTITION BY m.mode_description ORDER BY t.travel_distance"
            "         ) AS decile "
            "  FROM Trip t JOIN Mode m ON t.mode = m.mode_id "
            "  WHERE t.travel_distance > 0"
            ") GROUP BY mode, decile"
        ),
        "target": str(target),
        "target_key_cols": ["mode", "decile"],
        "target_value_col": "distance",
        "obs_noise_std": obs_noise_std,
        "model_discrepancy_std": model_discrepancy_std,
        "weight_per_element": weight_per_element,
        "aggregation": aggregation,
    }


# ----- Deliberately NOT shipped -----
#
# `boarding_by_agency` was published in v0.30 and REMOVED in v0.36: it
# queried `MM_Trip.agency`, and neither that column nor any other
# agency identifier exists anywhere in the polarislib Demand schema
# (`MM_Trip` is the micromobility table — bike/scooter — not transit
# boardings). Agency-level boarding counts require joining the Supply
# database's transit tables, which is outside what a `demand_db`-keyed
# `moment_set` moment can reach. Users who need that moment should
# write a project-local `source_sql` against an ATTACH-ed Supply DB, or
# pre-aggregate boardings into a small CSV and score it with a separate
# metric.

__all__ = [
    "mean_travel_time_by_activity",
    "mode_shares_by_purpose",
    "trip_distance_deciles_by_mode",
    "trip_mode_shares",
]

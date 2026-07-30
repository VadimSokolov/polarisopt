"""Reusable moment builders for :class:`polarisopt.metrics.MomentSetMetric`.

v0.30 P6. Each helper returns a canned ``moment`` dict that can be
dropped straight into a ``moment_set.moments`` YAML list, so users
don't re-implement the same SQL every study. The set here covers the
"POM starter kit" — the categorical patterns that Grimm et al. 2005
identifies as most productive for agent-based calibration:

- Mode shares by trip purpose × mode
- Boarding counts by transit agency × mode type
- Mean travel time by destination-activity type
- Trip-distance CDF (approximated by deciles) by mode

Each helper is a pure Python function that builds a moment dict; the
actual SQL runs at study time inside :class:`MomentSetMetric`. The
helpers assume the polarislib POLARIS Demand DB schema (Trip table,
MM_Trip table, Activity table with the standard columns).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def mode_shares_by_purpose(
    *,
    target: Path | str,
    obs_noise_std: float = 0.005,
    model_discrepancy_std: float = 0.02,
    weight_per_element: float = 1.0,
    aggregation: str = "elementwise_residual",
    name: str = "mode_shares_by_purpose",
) -> dict[str, Any]:
    """POM moment #1: mode share table by (purpose, mode).

    Target CSV columns: ``purpose``, ``mode``, ``share``. Simulator SQL
    normalizes by the count of non-null-purpose trips so shares sum to
    1 within each purpose.

    Defaults align with the DFW β-calibration design's Table 1
    (obs=0.005, md=0.02 — POLARIS Phase 3B.0 measurements).
    """
    return {
        "name": name,
        "source_sql": (
            "SELECT purpose, mode, "
            "COUNT(*) * 1.0 / (SELECT COUNT(*) FROM Trip WHERE purpose IS NOT NULL) "
            "AS share "
            "FROM Trip WHERE purpose IS NOT NULL "
            "GROUP BY purpose, mode"
        ),
        "target": str(target),
        "target_key_cols": ["purpose", "mode"],
        "target_value_col": "share",
        "obs_noise_std": obs_noise_std,
        "model_discrepancy_std": model_discrepancy_std,
        "weight_per_element": weight_per_element,
        "aggregation": aggregation,
    }


def boarding_by_agency(
    *,
    target: Path | str,
    obs_noise_std: float = 0.10,
    model_discrepancy_std: float = 0.20,
    weight_per_element: float = 1.0,
    aggregation: str = "log_ratio_residual",
    name: str = "boarding_by_agency",
) -> dict[str, Any]:
    """POM moment #2: transit boarding count table by (agency, mode_type).

    Target CSV columns: ``agency``, ``type``, ``boardings``. Uses
    ``log_ratio_residual`` by default — boarding counts span multiple
    orders of magnitude across agencies and elementwise residuals over-
    weight the big-agency buckets. Defaults obs=0.10, md=0.20 encode
    the "10% obs / 20% model" ratio the DFW study uses.
    """
    return {
        "name": name,
        "source_sql": (
            "SELECT agency, type, COUNT(*) * 1.0 AS boardings "
            "FROM MM_Trip WHERE agency IS NOT NULL "
            "GROUP BY agency, type"
        ),
        "target": str(target),
        "target_key_cols": ["agency", "type"],
        "target_value_col": "boardings",
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
    """POM moment #3: mean travel time (minutes) by destination-activity type.

    Target CSV columns: ``dest_act_type``, ``mean_travel_time``. Obs
    noise units are minutes (defaults obs=2 min, md=4 min from DFW).
    """
    return {
        "name": name,
        "source_sql": (
            "SELECT dest.type AS dest_act_type, "
            "AVG(Trip.travel_time) AS mean_travel_time "
            "FROM Trip JOIN Activity dest ON Trip.destination = dest.activity_id "
            "WHERE dest.type IS NOT NULL "
            "GROUP BY dest.type"
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
    """POM moment #4: trip-distance deciles by mode.

    Target CSV columns: ``mode``, ``decile`` (1..10), ``distance``.
    The SQL uses ``NTILE(10) OVER (PARTITION BY mode ORDER BY travel_distance)``
    and takes the max per (mode, decile) as the decile boundary. This is
    a coarse but robust approximation of the empirical CDF.

    Defaults units are miles (obs=0.5 mi, md=1 mi from DFW).
    """
    return {
        "name": name,
        "source_sql": (
            "SELECT mode, decile, MAX(travel_distance) AS distance FROM ("
            "  SELECT mode, travel_distance, "
            "  NTILE(10) OVER (PARTITION BY mode ORDER BY travel_distance) AS decile "
            "  FROM Trip WHERE travel_distance > 0"
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


__all__ = [
    "boarding_by_agency",
    "mean_travel_time_by_activity",
    "mode_shares_by_purpose",
    "trip_distance_deciles_by_mode",
]

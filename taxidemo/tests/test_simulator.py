"""Unit tests for the pure-Python taxi simulator."""

import math

import pytest

from taxidemo.simulator import OUTPUT_KEYS, TaxiParams, TaxiSimulation, run_taxi_simulation


def test_outputs_have_expected_keys():
    out = run_taxi_simulation(seed=0, max_steps=200)
    assert set(out) == set(OUTPUT_KEYS)
    assert all(math.isfinite(v) for v in out.values())


def test_same_seed_is_deterministic():
    a = run_taxi_simulation(seed=123)
    b = run_taxi_simulation(seed=123)
    assert a == b


def test_different_seeds_differ():
    a = run_taxi_simulation(seed=1)
    b = run_taxi_simulation(seed=2)
    assert a != b


def test_zero_demand_yields_pure_operating_cost():
    # No journeys: profit is exactly the per-step operating cost, charged on
    # every step except the final one.
    out = run_taxi_simulation(seed=0, journey_frequency=0, taxi_count=20, max_steps=1000)
    assert out["journeys_completed"] == 0
    assert out["missed"] == 0
    assert out["profit"] == pytest.approx(-0.1 * 20 * 999)
    assert out["profit_per_journey"] == 0.0


def test_high_prices_cancel_most_journeys():
    out = run_taxi_simulation(seed=0, base_fare=15, cost_per_tile=20, max_multiplier=3)
    assert out["missed"] > 300
    assert out["journeys_completed"] < out["missed"]


def test_more_taxis_complete_more_journeys():
    few = sum(run_taxi_simulation(seed=s, taxi_count=5)["journeys_completed"] for s in range(3))
    many = sum(run_taxi_simulation(seed=s, taxi_count=60)["journeys_completed"] for s in range(3))
    assert many > few


def test_profit_decomposition_is_consistent():
    # profit == sum(fares) - operating cost, and profit_per_journey excludes the cost.
    p = TaxiParams()
    sim = TaxiSimulation(p, seed=0)
    out = sim.run()
    operating_cost = 0.1 * p.taxi_count * (p.max_steps - 1)
    gross = out["profit_per_journey"] * out["journeys_completed"]
    assert gross == pytest.approx(out["profit"] + operating_cost)


def test_outstanding_journeys_respect_cap():
    p = TaxiParams(taxi_count=1, journey_frequency=100)
    sim = TaxiSimulation(p, seed=0)
    cap = sim.max_journeys
    while sim.clock < p.max_steps:
        sim.step()
        assert len(sim.journeys) <= cap


def test_grid_geometry():
    sim = TaxiSimulation(TaxiParams(grid_size=5), seed=0)
    assert sim.grid_width == 52
    assert sim.max_journeys == 160
    # Lane directions: x%12==1 flows +z, x%12==2 flows -z (away from edges).
    assert sim.valid_moves((13, 20)) == (0, 1)
    assert sim.valid_moves((14, 20)) == (0, -1)
    # z%12==1 flows -x, z%12==2 flows +x.
    assert sim.valid_moves((20, 13)) == (-1, 0)
    assert sim.valid_moves((20, 14)) == (1, 0)


def test_bfs_follows_one_way_lanes():
    sim = TaxiSimulation(TaxiParams(grid_size=5), seed=0)
    origin, destination = (13, 20), (14, 30)
    path = sim.bfs(origin, destination)
    assert path[-1] == destination
    assert origin not in path
    # Each hop must be a legal single-tile move.
    prev = origin
    for pos in path:
        vx, vz = sim.valid_moves(prev)
        assert pos in ((prev[0] + vx, prev[1]), (prev[0], prev[1] + vz))
        prev = pos


def test_adjacent_road_tile():
    sim = TaxiSimulation(TaxiParams(grid_size=5), seed=0)
    # Sidewalk at x%12==0 maps one tile east; x%12==3 one tile west.
    assert sim.adjacent_road_tile((12, 20)) == (13, 20)
    assert sim.adjacent_road_tile((15, 20)) == (14, 20)
    assert sim.adjacent_road_tile((20, 12)) == (20, 13)
    assert sim.adjacent_road_tile((20, 15)) == (20, 14)

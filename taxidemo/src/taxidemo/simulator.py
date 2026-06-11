"""Pure-Python port of the Emukit Playground taxi simulator.

A faithful re-implementation of ``js/simulators/taxi.js`` from
https://github.com/amzn/emukit-playground (Apache-2.0), with the Three.js
rendering stripped out and the RNG made seedable.

The city is a square grid of one-way two-lane roads. Taxis drive one tile
per simulation step; journeys spawn on sidewalks at a configurable rate,
are dispatched to the nearest available taxi (straight-line distance), and
pay ``base_fare + duration * (1 + q^2 * (max_multiplier - 1)) * cost_per_tile / 10``
on completion, where ``q`` is the fraction of the demand backlog in use.
Each step also charges an operating cost of ``0.1 * taxi_count``.

Intentional fidelity notes (quirks of the original kept on purpose):

- The journey-cancellation formula uses the playground's *slider maxima*
  (base fare 15, cost per km 20, multiplier 3) as constants, regardless of
  the current parameter values.
- Queued journeys are only re-dispatched when a taxi finishes a route and
  becomes available.
- A taxi stuck behind traffic for 3+ steps sidesteps to any free lane tile
  and re-plans its route with BFS.
- Journeys still pending (queued, awaiting pickup, or in transit) when the
  clock reaches ``max_steps`` all count as missed customers.
- The per-step operating cost is not charged on the final step.

The one deliberate deviation: in the original, a price-cancelled journey
briefly lingers in the journey list (it is removed by a wall-clock timer),
which can slightly depress the spawn cap in heavy-cancellation regimes.
Here cancelled journeys never enter the list.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Optional

ROAD_WIDTH = 2
ROAD_LENGTH = 10
ROAD_SIZE = ROAD_WIDTH + ROAD_LENGTH  # 12

# Playground slider maxima — fixed constants in the cancellation formula.
BASE_FARE_SLIDER_MAX = 15.0
COST_PER_TILE_SLIDER_MAX = 20.0
MAX_MULTIPLIER_SLIDER_MAX = 3.0

OUTPUT_KEYS = ("profit", "missed", "pick_up_time", "journeys_completed", "profit_per_journey")


@dataclass
class TaxiParams:
    """Simulation inputs, with the playground's defaults and ranges.

    grid_size:         number of roads in each direction (3..10)
    taxi_count:        fleet size (1..100)
    journey_frequency: journeys spawned per 10 steps (1..100)
    base_fare:         flag-drop fare in $ (0..15)
    cost_per_tile:     price per km; per-step ride revenue = value / 10 (1..20)
    max_multiplier:    surge-pricing cap (1..3)
    max_steps:         simulation length in steps (playground uses 1000)
    """

    grid_size: int = 5
    taxi_count: int = 20
    journey_frequency: float = 50.0
    base_fare: float = 5.0
    cost_per_tile: float = 5.0
    max_multiplier: float = 2.0
    max_steps: int = 1000

    def __post_init__(self) -> None:
        self.grid_size = int(self.grid_size)
        self.taxi_count = int(self.taxi_count)
        self.max_steps = int(self.max_steps)
        if not 2 <= self.grid_size:
            raise ValueError(f"grid_size must be >= 2, got {self.grid_size}")
        if self.taxi_count < 1:
            raise ValueError(f"taxi_count must be >= 1, got {self.taxi_count}")
        if self.journey_frequency < 0:
            raise ValueError(f"journey_frequency must be >= 0, got {self.journey_frequency}")


class _Taxi:
    __slots__ = ("sim", "position", "available", "path", "stopped", "cb")

    def __init__(self, sim: "TaxiSimulation", position: tuple[int, int]) -> None:
        self.sim = sim
        self.position = position
        self.available = True
        self.path: list[tuple[int, int]] = []
        self.stopped = 0
        self.cb: Optional[Callable[[], None]] = None

    def goto(self, destination: tuple[int, int], cb: Callable[[], None]) -> None:
        self.path = self.sim.bfs(self.position, destination)
        self.cb = cb
        if self.position == destination:
            self.finish_route()

    def finish_route(self) -> None:
        if self.cb is not None:
            self.cb()
        if self.available and self.sim.queue:
            self.sim.queue[0].search()

    def set_position(self, new_position: tuple[int, int]) -> bool:
        if new_position in self.sim.occupied:
            return False
        self.stopped = 0
        self.sim.occupied.discard(self.position)
        self.sim.occupied.add(new_position)
        self.position = new_position
        return True

    def update(self) -> None:
        sim = self.sim
        if self.path:
            if self.set_position(self.path[0]):
                self.path.pop(0)
                if not self.path:
                    self.finish_route()
            elif self.stopped >= 3:
                # Stuck in traffic: sidestep to any free lane tile, then re-plan.
                vx, vz = sim.valid_moves(self.position)
                moved = self.set_position((self.position[0] + vx, self.position[1]))
                if not moved:
                    moved = self.set_position((self.position[0], self.position[1] + vz))
                if moved:
                    self.path = sim.bfs(self.position, self.path[-1])
            else:
                self.stopped += 1
        else:
            # Idle wandering: follow the lane; choose uniformly at intersections.
            vx, vz = sim.valid_moves(self.position)
            move_x = True
            if vx == 0:
                move_x = False
            elif vz != 0:
                move_x = sim.rng.random() >= 0.5
            if move_x:
                self.set_position((self.position[0] + vx, self.position[1]))
            else:
                self.set_position((self.position[0], self.position[1] + vz))


class _Journey:
    __slots__ = ("sim", "origin", "destination", "start_time", "pickup_time", "taxi")

    def __init__(self, sim: "TaxiSimulation", origin: tuple[int, int], destination: tuple[int, int]) -> None:
        self.sim = sim
        self.origin = origin
        self.destination = destination
        self.start_time = sim.clock
        self.pickup_time: Optional[int] = None
        self.taxi: Optional[_Taxi] = None

    def search(self) -> None:
        sim = self.sim
        # In the original this always removes the queue head; in practice
        # search() is only ever re-triggered on the head, so this is the same.
        if self in sim.queue:
            sim.queue.remove(self)
        best: Optional[_Taxi] = None
        best_dist = math.inf
        for taxi in sim.taxis:
            if not taxi.available:
                continue
            dist = math.dist(taxi.position, self.origin)
            if dist < best_dist:
                best_dist = dist
                best = taxi
        if best is None:
            sim.queue.append(self)
            return
        self.taxi = best
        best.available = False
        best.goto(sim.adjacent_road_tile(self.origin), self._pickup)

    def _pickup(self) -> None:
        sim = self.sim
        sim.occupied.discard(self.origin)
        assert self.taxi is not None
        self.taxi.goto(sim.adjacent_road_tile(self.destination), self._end)
        self.pickup_time = sim.clock
        wait = self.pickup_time - self.start_time
        sim.pick_up_time = (sim.pick_up_time * sim.pickups_completed + wait) / (sim.pickups_completed + 1)
        sim.pickups_completed += 1

    def _end(self) -> None:
        sim = self.sim
        sim.occupied.discard(self.destination)
        assert self.taxi is not None and self.pickup_time is not None
        self.taxi.available = True
        duration = sim.clock - self.pickup_time
        pressure = len(sim.queue) / sim.max_journeys
        extra = duration * (pressure * pressure) * (sim.params.max_multiplier - 1)
        fare = sim.params.base_fare + (duration + extra) * (sim.params.cost_per_tile / 10)
        sim.profit += fare
        sim.gross_profit += fare
        sim.journeys_completed += 1
        sim.journeys.remove(self)

    def dispose(self) -> None:
        sim = self.sim
        if self.taxi is not None:
            self.taxi.available = True
            self.taxi.path = []
        sim.missed += 1
        sim.occupied.discard(self.origin)
        sim.occupied.discard(self.destination)
        if self in sim.journeys:
            sim.journeys.remove(self)
        if self in sim.queue:
            sim.queue.remove(self)


class TaxiSimulation:
    """One seeded simulation run. Use :func:`run_taxi_simulation` for one-shot use."""

    def __init__(self, params: TaxiParams, seed: Optional[int] = None) -> None:
        self.params = params
        self.rng = random.Random(seed)
        self.grid_width = ROAD_WIDTH * params.grid_size + ROAD_LENGTH * (params.grid_size - 1) + 2  # 12g - 8
        self.max_journeys = (params.grid_size - 1) * 4 * ROAD_LENGTH

        self.occupied: set[tuple[int, int]] = set()
        self.journeys: list[_Journey] = []
        self.queue: list[_Journey] = []

        self.clock = 0
        self._call_tick = 0
        self._carry = 0.0

        self.profit = 0.0
        self.gross_profit = 0.0
        self.missed = 0
        self.pick_up_time = 0.0
        self.pickups_completed = 0
        self.journeys_completed = 0

        self.taxis = [_Taxi(self, self._random_road()) for _ in range(params.taxi_count)]

    # -- geometry ---------------------------------------------------------

    def valid_moves(self, position: tuple[int, int]) -> tuple[int, int]:
        x, z = position
        xm, zm = x % ROAD_SIZE, z % ROAD_SIZE
        vx = vz = 0
        if xm == 1:
            vz += 1
        if xm == 2:
            vz -= 1
        if zm == 1:
            vx -= 1
        if zm == 2:
            vx += 1
        # Don't drive off the edge of the grid.
        if (x == 1 and vx == -1) or (x == self.grid_width - 2 and vx == 1):
            vx = 0
        if (z == 1 and vz == -1) or (z == self.grid_width - 2 and vz == 1):
            vz = 0
        return vx, vz

    @staticmethod
    def adjacent_road_tile(sidewalk: tuple[int, int]) -> tuple[int, int]:
        x, z = sidewalk
        if x % ROAD_SIZE == 0:
            x += 1
        if x % ROAD_SIZE == ROAD_WIDTH + 1:
            x -= 1
        if z % ROAD_SIZE == 0:
            z += 1
        if z % ROAD_SIZE == ROAD_WIDTH + 1:
            z -= 1
        return x, z

    def bfs(self, origin: tuple[int, int], destination: tuple[int, int]) -> list[tuple[int, int]]:
        """Shortest path over the directed lane graph, excluding origin, including destination."""
        if origin == destination:
            return []
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        visited = {origin}
        frontier: deque[tuple[int, int]] = deque([origin])
        while frontier:
            pos = frontier.popleft()
            if pos == destination:
                path = []
                while pos != origin:
                    path.append(pos)
                    pos = parent[pos]
                path.reverse()
                return path
            vx, vz = self.valid_moves(pos)
            if vx != 0:
                nxt = (pos[0] + vx, pos[1])
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = pos
                    frontier.append(nxt)
            if vz != 0:
                nxt = (pos[0], pos[1] + vz)
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = pos
                    frontier.append(nxt)
        return []

    # -- random placement ---------------------------------------------------

    def _rand(self, max_: int, min_: int = 0) -> int:
        return math.floor(min_ + self.rng.random() * (max_ - min_))

    def _random_position(self, lateral_offsets: tuple[int, int]) -> tuple[int, int]:
        g = self.params.grid_size
        while True:
            p1 = ROAD_SIZE * self._rand(g) + lateral_offsets[self._rand(2)]
            p2 = 3 + ROAD_SIZE * self._rand(g - 1) + self._rand(ROAD_LENGTH)
            position = (p1, p2) if self._rand(2) == 0 else (p2, p1)
            if position not in self.occupied:
                self.occupied.add(position)
                return position

    def _random_sidewalk(self) -> tuple[int, int]:
        return self._random_position((0, 3))

    def _random_road(self) -> tuple[int, int]:
        return self._random_position((1, 2))

    # -- demand --------------------------------------------------------------

    def _event_occurs(self, value: float, max_: float) -> bool:
        return value / max_ > self.rng.random()

    def _spawn_journey(self) -> None:
        origin = self._random_sidewalk()
        destination = self._random_sidewalk()
        p = self.params
        b_half = BASE_FARE_SLIDER_MAX / 2
        c_half = COST_PER_TILE_SLIDER_MAX / 2
        c_discount = 1 - (p.cost_per_tile - c_half) / b_half
        m_pressure = 0.9 * p.max_multiplier / MAX_MULTIPLIER_SLIDER_MAX
        cancel = (
            p.base_fare > b_half and self._event_occurs(p.base_fare - b_half + c_discount * m_pressure, b_half)
        ) or (
            p.cost_per_tile > c_half and self._event_occurs(p.cost_per_tile - c_half + c_discount * m_pressure, c_half)
        )
        if cancel:
            self.missed += 1
            self.occupied.discard(origin)
            self.occupied.discard(destination)
            return
        journey = _Journey(self, origin, destination)
        self.journeys.append(journey)
        journey.search()

    # -- main loop -------------------------------------------------------------

    def step(self) -> None:
        for taxi in self.taxis:
            taxi.update()
        self.clock += 1
        if self.clock >= self.params.max_steps:
            return
        self._call_tick += 1
        if self._call_tick >= 10:
            self._call_tick = 0
            rate = self.params.journey_frequency / 10
            whole = math.floor(rate)
            self._carry += rate - whole
            while len(self.journeys) < self.max_journeys and whole > 0:
                self._spawn_journey()
                whole -= 1
            while len(self.journeys) < self.max_journeys and self._carry > 1:
                self._spawn_journey()
                self._carry -= 1
        self.profit -= 0.1 * self.params.taxi_count

    def run(self) -> dict[str, float]:
        while self.clock < self.params.max_steps:
            self.step()
        for journey in list(self.journeys):
            journey.dispose()
        return self.outputs()

    def outputs(self) -> dict[str, float]:
        completed = self.journeys_completed
        return {
            "profit": self.profit,
            "missed": float(self.missed),
            "pick_up_time": self.pick_up_time,
            "journeys_completed": float(completed),
            "profit_per_journey": self.gross_profit / completed if completed else 0.0,
        }


def run_taxi_simulation(seed: Optional[int] = None, **params: float) -> dict[str, float]:
    """Run one simulation and return its output metrics.

    Parameters are the fields of :class:`TaxiParams` (any subset; the rest
    take the playground defaults). ``seed`` makes the run reproducible.

    >>> out = run_taxi_simulation(seed=0, taxi_count=20)
    >>> sorted(out) == sorted(OUTPUT_KEYS)
    True
    """
    return TaxiSimulation(TaxiParams(**params), seed=seed).run()


def default_params() -> dict[str, float]:
    """The playground's default inputs as a plain dict."""
    return asdict(TaxiParams())

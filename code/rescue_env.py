"""Rescue robot environment for HW3.

This file is part of the starter code and should not be modified by students.
The environment is deliberately small and tabular so finite-horizon DP and
tabular RL experiments run comfortably on a normal laptop CPU.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable

from rescue_types import (
    ACTIONS,
    ACTION_TO_DELTA,
    HAZARD_TRANSITIONS,
    LEFT_OF,
    MOVE_ACTIONS,
    RIGHT_OF,
    Action,
    State,
)


class RescueEnv:
    """A small finite-horizon stochastic grid MDP.

    State format:
        (row, col, carrying_sensor, battery_level, time_remaining)

    Process rewards are returned on transitions. Terminal outcomes are encoded
    as rewards when entering a terminal state.
    """

    def __init__(
        self,
        map_path: str | Path,
        hazard_path: str | Path,
        config_path: str | Path,
    ) -> None:
        self.map_path = Path(map_path)
        self.hazard_path = Path(hazard_path)
        self.config_path = Path(config_path)
        self.grid = self._read_csv_grid(self.map_path)
        self.hazards = self._read_csv_grid(self.hazard_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.height = len(self.grid)
        self.width = len(self.grid[0])
        if len(self.hazards) != self.height or any(len(row) != self.width for row in self.hazards):
            raise ValueError("hazard map shape must match rescue map shape")

        self.max_battery = int(self.config["max_battery"])
        self.default_horizon = int(self.config["default_horizon"])
        self.step_cost = float(self.config["step_cost"])
        self.success_reward = float(self.config["success_reward"])
        self.danger_reward = float(self.config["danger_reward"])
        self.timeout_penalty = float(self.config["timeout_penalty"])
        self.battery_empty_penalty = float(self.config["battery_empty_penalty"])
        self.recharge_amount = int(self.config["recharge_amount"])

        self.start_pos = self._find_unique("S")
        self.pickup_pos = self._find_unique("P")

        self._rng = random.Random()
        self._current_state: State | None = None

    @staticmethod
    def _read_csv_grid(path: Path) -> list[list[str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = [[cell.strip() for cell in row] for row in csv.reader(f)]
        if not rows or not rows[0]:
            raise ValueError(f"{path} is empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError(f"{path} must be rectangular")
        return rows

    def _find_unique(self, symbol: str) -> tuple[int, int]:
        locations = [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, value in enumerate(row)
            if value == symbol
        ]
        if len(locations) != 1:
            raise ValueError(f"expected exactly one {symbol!r} cell, found {len(locations)}")
        return locations[0]

    def initial_state(self, horizon: int | None = None) -> State:
        return (
            self.start_pos[0],
            self.start_pos[1],
            False,
            self.max_battery,
            self.default_horizon if horizon is None else int(horizon),
        )

    def reset(self, seed: int | None = None, horizon: int | None = None) -> State:
        if seed is not None:
            self._rng.seed(seed)
        self._current_state = self.initial_state(horizon)
        return self._current_state

    def step(self, action: Action) -> tuple[State, float, bool, dict[str, Any]]:
        if self._current_state is None:
            self.reset()
        assert self._current_state is not None
        outcomes = self.transitions(self._current_state, action)
        if not outcomes:
            return self._current_state, 0.0, True, {"terminal": True}

        roll = self._rng.random()
        total = 0.0
        chosen = outcomes[-1]
        for probability, next_state, reward in outcomes:
            total += probability
            if roll <= total + 1e-12:
                chosen = (probability, next_state, reward)
                break

        _, next_state, reward = chosen
        self._current_state = next_state
        return next_state, reward, self.is_terminal(next_state), {}

    def cell(self, row: int, col: int) -> str:
        return self.grid[row][col]

    def hazard(self, row: int, col: int) -> str:
        hazard = self.hazards[row][col]
        if hazard == "#":
            return "wall"
        return hazard

    def is_wall(self, row: int, col: int) -> bool:
        return self.grid[row][col] == "#"

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_terminal(self, state: State) -> bool:
        row, col, carrying, battery, time_remaining = state
        cell = self.cell(row, col)
        return (
            (cell == "E" and carrying)
            or cell == "X"
            or battery <= 0
            or time_remaining <= 0
        )

    def actions(self, state: State) -> tuple[Action, ...]:
        if self.is_terminal(state):
            return ()
        row, col, *_ = state
        if self.cell(row, col) == "R":
            return ACTIONS
        return MOVE_ACTIONS

    def transitions(self, state: State, action: Action) -> list[tuple[float, State, float]]:
        if self.is_terminal(state):
            return []
        if action not in self.actions(state):
            raise ValueError(f"illegal action {action!r} in state {state!r}")

        row, col, carrying, battery, time_remaining = state
        hazard = self.hazard(row, col)
        primitive_moves = self._primitive_moves(action, hazard)

        merged: dict[tuple[State, float], float] = defaultdict(float)
        for move_action, probability in primitive_moves:
            next_row, next_col = self._apply_move(row, col, move_action)
            next_state, reward = self._finish_transition(
                next_row,
                next_col,
                carrying,
                battery,
                time_remaining,
                hazard,
            )
            merged[(next_state, reward)] += probability

        return [
            (probability, next_state, reward)
            for (next_state, reward), probability in sorted(
                merged.items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        ]

    def _primitive_moves(self, action: Action, hazard: str) -> list[tuple[Action, float]]:
        if action == "WAIT":
            return [("WAIT", 1.0)]
        if hazard not in HAZARD_TRANSITIONS:
            raise ValueError(f"unknown hazard label {hazard!r}")

        moves: list[tuple[Action, float]] = []
        for kind, probability in HAZARD_TRANSITIONS[hazard]:
            if kind == "intended":
                moves.append((action, probability))
            elif kind == "left":
                moves.append((LEFT_OF[action], probability))
            elif kind == "right":
                moves.append((RIGHT_OF[action], probability))
            elif kind == "stay":
                moves.append(("WAIT", probability))
            else:
                raise ValueError(f"unknown transition kind {kind!r}")
        return moves

    def _apply_move(self, row: int, col: int, action: Action) -> tuple[int, int]:
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = row + dr, col + dc
        if not self.in_bounds(nr, nc) or self.is_wall(nr, nc):
            return row, col
        return nr, nc

    def _finish_transition(
        self,
        row: int,
        col: int,
        carrying: bool,
        battery: int,
        time_remaining: int,
        source_hazard: str,
    ) -> tuple[State, float]:
        reward = self.step_cost
        time_next = max(0, time_remaining - 1)

        battery_cost = 2 if source_hazard == "damaged" else 1
        battery_next = max(0, battery - battery_cost)

        cell = self.cell(row, col)
        carrying_next = carrying or (row, col) == self.pickup_pos
        if cell == "R":
            battery_next = min(self.max_battery, battery_next + self.recharge_amount)

        next_state: State = (row, col, carrying_next, battery_next, time_next)

        if cell == "E" and carrying_next:
            reward += self.success_reward
        elif cell == "X":
            reward += self.danger_reward
        elif battery_next <= 0:
            reward += self.battery_empty_penalty
        elif time_next <= 0:
            reward += self.timeout_penalty

        return next_state, reward

    def reachable_states(self, horizon: int | None = None) -> list[State]:
        start = self.initial_state(horizon)
        seen = {start}
        queue: deque[State] = deque([start])
        while queue:
            state = queue.popleft()
            for action in self.actions(state):
                for _, next_state, _ in self.transitions(state, action):
                    if next_state not in seen:
                        seen.add(next_state)
                        queue.append(next_state)
        return sorted(seen)

    def simulate(
        self,
        policy: dict[Any, Any] | Callable[[State], Action],
        horizon: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        state = self.reset(seed=seed, horizon=horizon)
        total_return = 0.0
        length = 0
        trajectory = []

        while not self.is_terminal(state):
            action = self._policy_action(policy, state)
            next_state, reward, done, _ = self.step(action)
            trajectory.append((state, action, reward, next_state, done))
            total_return += reward
            length += 1
            state = next_state

        return {
            "return": total_return,
            "length": length,
            "terminal_reason": self.terminal_reason(state),
            "trajectory": trajectory,
        }

    def _policy_action(self, policy: dict[Any, Any] | Callable[[State], Action], state: State) -> Action:
        if callable(policy):
            return policy(state)
        time_remaining = state[4]
        if time_remaining in policy:
            action = policy[time_remaining].get(state)
            if action is not None:
                return action
            return self.actions(state)[0]
        action = policy.get(state)
        if action is not None:
            return action
        return self.actions(state)[0]

    def terminal_reason(self, state: State) -> str:
        row, col, carrying, battery, time_remaining = state
        cell = self.cell(row, col)
        if cell == "E" and carrying:
            return "success"
        if cell == "X":
            return "danger"
        if battery <= 0:
            return "battery"
        if time_remaining <= 0:
            return "timeout"
        return "non_terminal"


def state_to_jsonable(state: State) -> list[Any]:
    row, col, carrying, battery, time_remaining = state
    return [row, col, carrying, battery, time_remaining]


def state_from_jsonable(value: Iterable[Any]) -> State:
    row, col, carrying, battery, time_remaining = value
    return (int(row), int(col), bool(carrying), int(battery), int(time_remaining))


def load_default_env(root: str | Path | None = None) -> RescueEnv:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return RescueEnv(
        base / "data" / "rescue_map.csv",
        base / "data" / "hazard_map_known.csv",
        base / "data" / "rescue_config.json",
    )


def load_jsonl_transitions(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                record["state"] = state_from_jsonable(record["state"])
                record["next_state"] = state_from_jsonable(record["next_state"])
                records.append(record)
    return records

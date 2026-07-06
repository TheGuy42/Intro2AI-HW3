"""Generate quick visualization artifacts for the rescue robot starter code.

This script does not depend on any student TODO implementation. It builds a
simple hand-written policy, saves map and policy-slice images, and tries to save
one animated rollout.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rescue_env import RescueEnv, load_default_env
from rescue_types import ACTION_TO_DELTA, MOVE_ACTIONS, Action, State
from visualize import animate_trajectory, plot_map, save_policy_slice


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "demo_outputs"
    output_dir.mkdir(exist_ok=True)

    env = load_default_env(root)
    horizon = env.default_horizon
    policy = build_demo_policy(env, horizon)

    fig, _ax = plot_map(env)
    fig.savefig(output_dir / "rescue_map.png", dpi=180, bbox_inches="tight")

    save_policy_slice(
        env,
        policy,
        output_dir / "policy_slice_t5_carrying_b2.png",
        time_remaining=5,
        carrying=True,
        battery=2,
    )
    save_policy_slice(
        env,
        policy,
        output_dir / "policy_slice_t15_not_carrying_b1.png",
        time_remaining=15,
        carrying=False,
        battery=1,
    )

    result = env.simulate(policy, horizon=horizon, seed=236504)
    try:
        animate_trajectory(env, result["trajectory"], output_dir / "demo_trajectory.gif", interval=120)
        animation_message = "demo_trajectory.gif"
    except Exception as exc:  # pragma: no cover - depends on local image writers.
        animation_message = f"animation skipped ({exc})"

    print(f"wrote {output_dir / 'rescue_map.png'}")
    print(f"wrote {output_dir / 'policy_slice_t5_carrying_b2.png'}")
    print(f"wrote {output_dir / 'policy_slice_t15_not_carrying_b1.png'}")
    print(f"rollout return={result['return']:.1f}, length={result['length']}, terminal={result['terminal_reason']}")
    print(animation_message)


def build_demo_policy(env: RescueEnv, horizon: int) -> dict[int, dict[State, Action]]:
    """Return a simple time-dependent policy for demonstration images."""

    pickup = _find_cell(env, "P")
    exit_cell = _find_cell(env, "E")
    recharge = _find_cell(env, "R")
    policy: dict[int, dict[State, Action]] = {t: {} for t in range(1, horizon + 1)}

    for state in env.reachable_states(horizon):
        if env.is_terminal(state):
            continue
        row, col, carrying, battery, time_remaining = state
        target = pickup
        if carrying:
            target = recharge if battery <= 2 and (row, col) != recharge else exit_cell
        if (row, col) == recharge and battery < env.max_battery and time_remaining > 5:
            action = "WAIT"
        else:
            action = _first_step_toward(env, (row, col), target)
        policy[time_remaining][state] = action

    return policy


def _find_cell(env: RescueEnv, symbol: str) -> tuple[int, int]:
    for row, cells in enumerate(env.grid):
        for col, cell in enumerate(cells):
            if cell == symbol:
                return (row, col)
    raise ValueError(f"cell {symbol!r} not found")


def _first_step_toward(env: RescueEnv, start: tuple[int, int], target: tuple[int, int]) -> Action:
    if start == target:
        return "WAIT"

    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], Action]] = {}
    seen = {start}

    while queue:
        row, col = queue.popleft()
        for action in MOVE_ACTIONS:
            dr, dc = ACTION_TO_DELTA[action]
            nxt = (row + dr, col + dc)
            nr, nc = nxt
            if not env.in_bounds(nr, nc) or env.is_wall(nr, nc) or nxt in seen:
                continue
            parent[nxt] = ((row, col), action)
            if nxt == target:
                return _recover_first_action(start, target, parent)
            seen.add(nxt)
            queue.append(nxt)

    return "WAIT"


def _recover_first_action(
    start: tuple[int, int],
    target: tuple[int, int],
    parent: dict[tuple[int, int], tuple[tuple[int, int], Action]],
) -> Action:
    current = target
    first_action = "WAIT"
    while current != start:
        previous, action = parent[current]
        first_action = action
        current = previous
    return first_action


if __name__ == "__main__":
    main()

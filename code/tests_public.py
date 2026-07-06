"""Tiny public tests for HW3.

Run from the code directory:
    python tests_public.py

These tests are intentionally small. Passing them does not guarantee full
correctness on the assignment map or hidden tests.
"""

from __future__ import annotations

import math
import argparse
from pathlib import Path

from rescue_env import load_default_env, load_jsonl_transitions
from rescue_types import ACTIONS

import learning_rescue
import planning_rescue


def test_environment_loads() -> None:
    env = load_default_env(Path(__file__).resolve().parents[1])
    assert env.height == 6
    assert env.width == 6
    assert env.initial_state(12) == (0, 0, False, 6, 12)
    assert len(env.reachable_states(12)) < 800


def test_transition_probabilities_sum_to_one() -> None:
    env = load_default_env(Path(__file__).resolve().parents[1])
    for state in env.reachable_states(8):
        for action in env.actions(state):
            total = sum(prob for prob, _, _ in env.transitions(state, action))
            assert math.isclose(total, 1.0, abs_tol=1e-9), (state, action, total)


def test_offline_data_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    transitions = load_jsonl_transitions(root / "data" / "offline_rollouts_sparse_train.jsonl")
    assert len(transitions) >= 200
    first = transitions[0]
    assert set(first) == {"state", "action", "reward", "next_state", "done"}
    assert first["action"] in ACTIONS


def test_good_offline_data_has_success_signal() -> None:
    root = Path(__file__).resolve().parents[1]
    transitions = load_jsonl_transitions(root / "data" / "offline_rollouts_good_train.jsonl")
    assert len(transitions) >= 1000
    assert any(record["reward"] > 50 for record in transitions)
    assert any(record["done"] and record["reward"] > 50 for record in transitions)


def test_action_value_tiny_case() -> None:
    class TinyEnv:
        def transitions(self, state, action):
            assert state == "s"
            assert action == "a"
            return [
                (0.25, "x", 1.0),
                (0.75, "y", -1.0),
            ]

    value = planning_rescue.action_value(TinyEnv(), {"x": 4.0, "y": 2.0}, "s", "a")
    assert math.isclose(value, 0.25 * 5.0 + 0.75 * 1.0)


def test_finite_horizon_dp_tie_is_stable() -> None:
    env = load_default_env(Path(__file__).resolve().parents[1])
    _V, policy = planning_rescue.finite_horizon_dp(env, horizon=20)
    state = (3, 3, True, 2, 5)
    assert policy[5][state] == env.actions(state)[0]


def test_model_estimation_tiny_case() -> None:
    trajectories = [
        {"state": "s", "action": "UP", "reward": 1.0, "next_state": "x", "done": False},
        {"state": "s", "action": "UP", "reward": 3.0, "next_state": "x", "done": False},
        {"state": "s", "action": "UP", "reward": -1.0, "next_state": "y", "done": True},
    ]
    model = learning_rescue.estimate_transition_reward_model(trajectories, ("UP",), smoothing=1.0)
    assert set(model) == {"transitions", "actions_by_state", "states", "actions"}
    assert model["actions"] == ("UP",)
    assert model["actions_by_state"]["s"] == ("UP",)
    outcomes = model["transitions"][("s", "UP")]
    assert math.isclose(sum(probability for probability, _, _ in outcomes), 1.0)
    by_next_state = {next_state: (probability, reward) for probability, next_state, reward in outcomes}
    assert math.isclose(by_next_state["x"][0], 3 / 5)
    assert math.isclose(by_next_state["x"][1], 2.0)
    assert math.isclose(by_next_state["y"][0], 2 / 5)
    assert math.isclose(by_next_state["y"][1], -1.0)


INFRASTRUCTURE_TESTS = [
    test_environment_loads,
    test_transition_probabilities_sum_to_one,
    test_offline_data_loads,
    test_good_offline_data_has_success_signal,
]

IMPLEMENTATION_TESTS = [
    test_action_value_tiny_case,
    test_finite_horizon_dp_tie_is_stable,
    test_model_estimation_tiny_case,
]


def run_tests(tests) -> None:
    for test in tests:
        print(f"running {test.__name__}...")
        test()


def run_all(include_implementation: bool = True) -> None:
    tests = list(INFRASTRUCTURE_TESTS)
    if include_implementation:
        tests.extend(IMPLEMENTATION_TESTS)
    run_tests(tests)
    if include_implementation:
        print("all public tests passed")
    else:
        print("starter infrastructure tests passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infra",
        action="store_true",
        help="only test provided infrastructure; useful before implementing TODOs",
    )
    args = parser.parse_args()
    run_all(include_implementation=not args.infra)


if __name__ == "__main__":
    main()

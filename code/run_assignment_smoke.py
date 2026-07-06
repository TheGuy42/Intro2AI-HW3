"""Run a short end-to-end smoke test after implementing the TODO files.

This script is meant for students or staff who want to see the completed wet
assignment run in one place. It does not contain solution logic; it only calls
the functions implemented in planning_rescue.py and learning_rescue.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from learning_rescue import (
    boltzmann_action,
    constant_epsilon,
    decaying_epsilon,
    default_alpha_schedule,
    estimate_transition_reward_model,
    evaluate_rescue_agent,
    first_visit_mc,
    plan_with_estimated_model,
    q_learning_rescue,
    td_prediction,
)
from planning_rescue import finite_horizon_dp, rollout_time_dependent_policy
from rescue_env import load_default_env, load_jsonl_transitions
from rescue_types import ACTIONS
from visualize import animate_trajectory, save_policy_slice


SEED = 236501


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "smoke_outputs"
    output_dir.mkdir(exist_ok=True)

    env = load_default_env(root)
    print("HW3 rescue smoke run")
    print(f"map={env.height}x{env.width}, horizon={env.default_horizon}, max_battery={env.max_battery}")
    print()

    try:
        run_part_b(env, output_dir)
        # run_part_c(env, root, output_dir)
    except NotImplementedError:
        print("A TODO function still raises NotImplementedError.")
        print("Finish planning_rescue.py and learning_rescue.py, then run this script again.")
        raise SystemExit(1)

    print()
    print(f"wrote visual outputs to {output_dir}")
    print("smoke run completed")


def run_part_b(env: Any, output_dir: Path) -> dict[int, dict[Any, Any]]:
    print("Part B: true-model finite-horizon planner (oracle baseline)")
    policies: dict[int, dict[Any, Any]] = {}
    for horizon in (12, 20, 30):
        V, policy = finite_horizon_dp(env, horizon)
        start_value = V[horizon][env.initial_state(horizon)]
        stats = rollout_time_dependent_policy(env, policy, n_episodes=100, seed=SEED)
        policies[horizon] = policy
        print(
            f"  H={horizon:2d}: "
            f"states={len(env.reachable_states(horizon)):4d}, "
            f"V(start)={start_value:8.3f}, "
            f"success={_get(stats, 'success_rate'):.3f}, "
            f"mean_return={_get(stats, 'mean_return'):.3f}, "
            f"mean_length={_get(stats, 'mean_length'):.2f}"
        )

    policy = policies[20]
    save_policy_slice(
        env,
        policy,
        output_dir / "true_model_planner_policy_t5_carrying_b2.png",
        time_remaining=5,
        carrying=True,
        battery=2,
    )
    save_policy_slice(
        env,
        policy,
        output_dir / "true_model_planner_policy_t15_not_carrying_b1.png",
        time_remaining=15,
        carrying=False,
        battery=1,
    )
    result = env.simulate(policy, horizon=20, seed=236504)
    animate_trajectory(env, result["trajectory"], output_dir / "true_model_planner_rollout.gif", interval=120)
    print(f"  sample rollout: return={result['return']:.1f}, length={result['length']}, terminal={result['terminal_reason']}")
    print()
    return policy


def run_part_c(env: Any, root: Path, output_dir: Path) -> None:
    print("Part C: learned model, value prediction, and model-free control")
    sparse_model_policy = run_model_based_check(env, root, "offline_rollouts_sparse_train.jsonl", "sparse negative example", output_dir)
    run_model_based_check(env, root, "offline_rollouts_good_train.jsonl", "good coverage check", output_dir)

    _, oracle_policy = finite_horizon_dp(env, env.default_horizon)
    mc_values = first_visit_mc(env, oracle_policy, n_episodes=400, gamma=1.0, seed=SEED + 2)
    td_values = td_prediction(env, oracle_policy, n_episodes=400, alpha_schedule=default_alpha_schedule, gamma=1.0, seed=SEED + 3)
    common_states = sorted(set(mc_values) & set(td_values), key=repr)[:5]
    print("  MC vs TD sample values:")
    for state in common_states:
        print(f"    {state}: MC={mc_values[state]:8.3f}, TD={td_values[state]:8.3f}")

    for label, epsilon_schedule in (
        ("constant epsilon", constant_epsilon),
        ("decaying epsilon", decaying_epsilon),
    ):
        Q, learned_policy = q_learning_rescue(
            env,
            n_episodes=1000,
            alpha_schedule=default_alpha_schedule,
            epsilon_schedule=epsilon_schedule,
            gamma=1.0,
            seed=SEED + 4,
        )
        stats = evaluate_rescue_agent(env, learned_policy, n_episodes=150, seed=SEED + 5)
        print(
            f"  Q-learning ({label}): "
            f"policy_states={len(learned_policy):4d}, "
            f"Q_entries={len(Q):4d}, "
            f"success={_get(stats, 'success_rate'):.3f}, "
            f"mean_return={_get(stats, 'mean_return'):.3f}, "
            f"mean_length={_get(stats, 'mean_length'):.2f}"
        )

    start_actions = env.actions(env.initial_state())
    sampled = boltzmann_action({}, env.initial_state(), start_actions, temperature=1.0)
    print(f"  Boltzmann smoke call returned legal action: {sampled in start_actions}")

    save_policy_slice(
        env,
        sparse_model_policy,
        output_dir / "estimated_model_planner_sparse_policy_t11_carrying_b3.png",
        time_remaining=11,
        carrying=True,
        battery=3,
    )


def run_model_based_check(env: Any, root: Path, filename: str, label: str, output_dir: Path) -> dict[int, dict[Any, Any]]:
    records = load_jsonl_transitions(root / "data" / filename)
    model = estimate_transition_reward_model(records, ACTIONS, smoothing=0.1)
    model_policy = plan_with_estimated_model(model, env.initial_state(), env.default_horizon)
    model_stats = evaluate_rescue_agent(env, model_policy, n_episodes=100, seed=SEED + 1)
    success_events = sum(1 for record in records if record["done"] and record["reward"] > 50)
    print(
        f"  estimated-model planner ({label}, {filename}): "
        f"records={len(records)}, success_events={success_events}, "
        f"success={_get(model_stats, 'success_rate'):.3f}, "
        f"mean_return={_get(model_stats, 'mean_return'):.3f}, "
        f"mean_length={_get(model_stats, 'mean_length'):.2f}"
    )
    if "good" in filename:
        save_policy_slice(
            env,
            model_policy,
            output_dir / "estimated_model_planner_good_policy_t11_carrying_b3.png",
            time_remaining=11,
            carrying=True,
            battery=3,
        )
    return model_policy


def _get(stats: dict[str, Any], key: str) -> float:
    value = stats.get(key)
    if value is None:
        raise KeyError(f"expected key {key!r} in stats dictionary, got {sorted(stats)}")
    return float(value)


if __name__ == "__main__":
    main()

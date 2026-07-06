"""Student implementation file for HW3 Part C.

Only edit this file unless course staff explicitly instruct otherwise.
"""

from __future__ import annotations

from typing import Any, Callable

from rescue_env import RescueEnv
from rescue_types import Action, State

AlphaSchedule = Callable[[int], float]
EpsilonSchedule = Callable[[int], float]


def estimate_transition_reward_model(
    trajectories: list[dict[str, Any]],
    actions: tuple[Action, ...],
    smoothing: float,
) -> dict[str, Any]:
    """Estimate P_hat and R_hat from offline transitions.

    The input trajectories list contains dictionaries with:
        state, action, reward, next_state, done

    Use Laplace smoothing for transition probabilities over the next states
    observed for each (state, action) pair. For each observed
    (state, action, next_state), use the mean observed reward for that triple.

    Return exactly this dictionary schema:
        {
            "transitions": {
                (state, action): [(probability, next_state, mean_reward), ...],
                ...
            },
            "actions_by_state": {
                state: (action1, action2, ...),
                ...
            },
            "states": (state1, state2, ...),
            "actions": actions,
        }

    Store actions in the same order as the actions argument. Store transition
    lists in a deterministic order, for example sorted by repr(next_state).
    States or actions with no observations should not invent transitions.
    """

    raise NotImplementedError


def plan_with_estimated_model(
    model: dict[str, Any],
    start_state: State,
    horizon: int,
) -> dict[int, dict[State, Action]]:
    """Run finite-horizon planning on the learned model.

    Use the model schema returned by estimate_transition_reward_model. A state
    with no actions in actions_by_state should be treated as terminal with
    value 0. Break exact action-value ties according to the action order stored
    in model["actions"].
    """

    raise NotImplementedError


def first_visit_mc(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    n_episodes: int,
    gamma: float,
    seed: int,
) -> dict[State, float]:
    """Estimate V^pi from sampled episodes using first-visit Monte Carlo.

    Use random.Random(seed) to generate a fresh deterministic seed for each
    episode; do not pass the same seed to every rollout.
    """

    raise NotImplementedError


def td_prediction(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    n_episodes: int,
    alpha_schedule: AlphaSchedule,
    gamma: float,
    seed: int,
) -> dict[State, float]:
    """Estimate V^pi from online interaction using TD(0).

    Count TD updates from t=0 upward when calling alpha_schedule(t). Use a fresh
    deterministic episode seed derived from seed for each episode.
    """

    raise NotImplementedError


def q_learning_rescue(
    env: RescueEnv,
    n_episodes: int,
    alpha_schedule: AlphaSchedule,
    epsilon_schedule: EpsilonSchedule,
    gamma: float,
    seed: int,
) -> tuple[dict[tuple[State, Action], float], dict[State, Action]]:
    """Learn a tabular Q policy with epsilon-greedy exploration.

    Count Q updates from t=0 upward when calling alpha_schedule(t) and
    epsilon_schedule(t). During exploration, choose uniformly from all legal
    env.actions(state). Break greedy ties using env.actions(state) order.
    """

    raise NotImplementedError


def boltzmann_action(
    Q: dict[tuple[State, Action], float],
    state: State,
    actions: tuple[Action, ...],
    temperature: float,
) -> Action:
    """Sample an action according to softmax(Q(state, action) / temperature).

    Use the Python random module for sampling. Subtract the largest scaled
    Q-value before exponentiating for numerical stability.
    """

    raise NotImplementedError


def evaluate_rescue_agent(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Return success rate, mean return, mean length, and failure breakdown.

    Use random.Random(seed) to generate a fresh deterministic seed for each
    evaluation episode. If a learned policy has no action for a reachable state,
    the provided environment uses its first legal action as a deterministic
    fallback.
    """

    raise NotImplementedError


def default_alpha_schedule(t: int) -> float:
    """Alpha schedule required by the assignment."""

    return 0.5 / (1 + (t // 1000))


def constant_epsilon(_: int) -> float:
    return 0.1


def decaying_epsilon(t: int) -> float:
    return max(0.02, 1 / ((t + 1) ** 0.5))

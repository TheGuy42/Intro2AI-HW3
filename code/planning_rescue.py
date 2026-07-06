"""Student implementation file for HW3 Part B.

Only edit this file unless course staff explicitly instruct otherwise.
"""

from __future__ import annotations

from typing import Any

from rescue_env import RescueEnv
from rescue_types import Action, State

# state is a tuple of: row, col, carrying, battery, time_remaining


def action_value(env: RescueEnv, V_next: dict[State, float], state: State, action: Action) -> float:
    """Return the one-step finite-horizon backup for one action.

    Formula:
        sum_{s'} P(s' | s,a) * (R(s,a,s') + V_next[s'])

    Missing states in V_next must be treated as having value 0.0. This is
    useful for terminal states and for tiny public tests.
    """

    transitions = env.transitions(state, action) # list of (probability, next_state, reward)
    value = 0.0
    for probability, next_state, reward in transitions:
        value += probability * (reward + V_next.get(next_state, 0.0))
    return value


def finite_horizon_dp(env: RescueEnv, horizon: int) -> tuple[dict[int, dict[State, float]], dict[int, dict[State, Action]]]:
    """Return V[t][s] and a time-dependent policy pi[t][s].

    The state already includes time_remaining. Use env.reachable_states(horizon)
    once, then group reachable states by s[4]. V[t] should contain states whose
    embedded time_remaining is t.

    V[0][s] should be 0 for every reachable state with no time remaining.
    For t >= 1, use the finite-horizon Bellman recursion over env.actions(s).
    Terminal states should have value 0 and should not appear in the policy.

    If two actions have exactly the same value, choose the action that appears
    earlier in env.actions(s). This makes public and hidden tests deterministic.

    The returned policy should contain entries for t=1,...,horizon.
    """

    reachable_states = env.reachable_states(horizon)
    states_by_time = {t: [] for t in range(horizon + 1)}
    for state in reachable_states:
        time_remaining = state[4]
        states_by_time[time_remaining].append(state)

    V = {t: {} for t in range(horizon + 1)}
    Policy = {t: {} for t in range(1, horizon + 1)}

    for t in range(horizon + 1):
        for state in states_by_time[t]:
            if t == 0:
                V[t][state] = 0.0
            elif env.is_terminal(state):
                V[t][state] = 0.0
            else:
                best_value = float('-inf')
                best_action = None
                for action in env.actions(state): # calculating the action maximizing the value
                    value = action_value(env, V[t - 1], state, action)
                    if value > best_value:
                        best_value = value
                        best_action = action
                
                V[t][state] = best_value
                if best_action is not None:
                    Policy[t][state] = best_action
    
    return V, Policy
    

def rollout_time_dependent_policy(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]],
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Simulate a time-dependent policy and return summary statistics.

    Use random.Random(seed) to generate a fresh deterministic seed for each
    episode; do not pass the same seed to every rollout.

    Expected return keys:
        mean_return, success_rate, mean_length, failure_breakdown, returns
    """

    import random #TODO: ask in piazza if this is allowed
    rng = random.Random(seed)
    #TODO: no given horizon..
    
    total_return = 0.0
    total_successes = 0
    total_length = 0
    failure_breakdown = {}
    returns = []

    for episode in range(n_episodes):
        rng_seed = rng.randint(0, 2**32 - 1)  # Generate a new seed for each episode
        # print(f"Episode {episode + 1}/{n_episodes} with seed {rng_seed}")
        simulation_res = env.simulate(policy, seed=rng_seed)

        total_return += simulation_res["return"]
        total_length += simulation_res["length"]
        terminal_reason = simulation_res["terminal_reason"]
        returns.append(simulation_res["return"])

        if terminal_reason == "success":
            total_successes += 1
        else:
            if terminal_reason not in failure_breakdown:
                failure_breakdown[terminal_reason] = 0
            failure_breakdown[terminal_reason] += 1

    mean_return = total_return / n_episodes
    success_rate = total_successes / n_episodes
    mean_length = total_length / n_episodes
    
    return {
        "mean_return": mean_return,
        "success_rate": success_rate,
        "mean_length": mean_length,
        "failure_breakdown": failure_breakdown,
        "returns": returns,
    }
        

def compare_policy_slices(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]],
    states_to_show: list[State],
) -> list[dict[str, Any]]:
    """Return a compact table of policy changes for selected states.

    Include only policy entries that exist for the supplied states. Each row can
    be a dictionary such as:
        {"state": state, "actions_by_time": {t: action, ...}}
    """

    table = []
    for state in states_to_show:
        actions_by_time = {}
        for t in sorted(policy.keys()):
            if state in policy[t]:
                actions_by_time[t] = policy[t][state]
        if actions_by_time:
            table.append({"state": state, "actions_by_time": actions_by_time})
    return table

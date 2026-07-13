"""Student implementation file for HW3 Part C.

Only edit this file unless course staff explicitly instruct otherwise.
"""

from __future__ import annotations

from typing import Any, Callable

from rescue_env import RescueEnv
from rescue_types import Action, State

import random # TODO: ask in piazza if this is allowed

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

    model = {
        "transitions": {},
        "actions_by_state": {},
        "states": set(),
        "actions": actions,
    }
    Q_hat = {} # dictionary to hold observed next states and rewards for each (state, action)

    for record in trajectories:
        state = record["state"]
        action = record["action"]
        next_state = record["next_state"]
        reward = record["reward"]

        model["states"].add(state)
        if state not in model["actions_by_state"]:
            model["actions_by_state"][state] = set()
        # We will order the actions later
        model["actions_by_state"][state].add(action)

        if (state, action) not in Q_hat:
            Q_hat[(state, action)] = {}
        if next_state not in Q_hat[(state, action)]:
            Q_hat[(state, action)][next_state] = []
        Q_hat[(state, action)][next_state].append(reward)

    # Now compute the transition probabilities and mean rewards
    for (state, action), next_states in Q_hat.items():
        total_count = sum(len(rewards) for rewards in next_states.values())
        num_next_states = len(next_states)
        smoothed_total_count = total_count + smoothing * num_next_states

        transitions_list = []
        for next_state, rewards in sorted(next_states.items(), key=lambda x: repr(x[0])):
            count = len(rewards)
            probability = (count + smoothing) / smoothed_total_count
            mean_reward = sum(rewards) / count
            transitions_list.append((probability, next_state, mean_reward))

        model["transitions"][(state, action)] = transitions_list
    
    # transform actions_by_state sets to tuples in the order of actions argument
    for state, actions_set in model["actions_by_state"].items():
        model["actions_by_state"][state] = tuple(a for a in actions if a in actions_set)
    
    model["states"] = tuple(sorted(model["states"], key=repr))
    
    return model


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

    Policy = {t: {} for t in range(1, horizon + 1)}

    states_by_time = {t: [] for t in range(horizon + 1)}
    current_states = {start_state}

    for t in range(horizon, -1, -1):
        states_by_time[t] = list(current_states)
        next_states = set()
        for state in current_states:
            actions = model["actions_by_state"].get(state, ())
            for action in actions:
                transitions = model["transitions"].get((state, action), [])
                for _, next_state, _ in transitions:
                    next_states.add(next_state)
        current_states = next_states

    V = {t: {} for t in range(horizon + 1)}

    for t in range(horizon + 1):
        for state in states_by_time[t]:
            if t == 0:
                V[t][state] = 0.0
            else:
                best_value = float('-inf')
                best_action = None
                actions = model["actions_by_state"].get(state, ())
                for action in actions:
                    transitions = model["transitions"].get((state, action), [])
                    value = sum(prob * (reward + V[t - 1].get(next_state, 0.0)) for prob, next_state, reward in transitions)
                    if value > best_value:
                        best_value = value
                        best_action = action
                V[t][state] = best_value
                if best_action is not None:
                    Policy[t][state] = best_action

    return Policy



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
    rng = random.Random(seed)

    V_pi = {}

    for episode_id in range(n_episodes):
        episode_seed = rng.randint(0, 2**32 - 1)
        episode = env.simulate(policy, seed=episode_seed)
        trajectory = episode["trajectory"]

        trajectory_reward = 0.0
        visited_states = set()
        reversed_trajectory = list(reversed(trajectory))
        for step in reversed_trajectory:
            state, action, reward, next_state, done = step
            trajectory_reward = reward + gamma * trajectory_reward
            if state not in visited_states:
                visited_states.add(state)
                if state not in V_pi:
                    V_pi[state] = []
                V_pi[state].append(trajectory_reward)
    
    # Average the returns for each state
    for state in V_pi:
        V_pi[state] = sum(V_pi[state]) / len(V_pi[state])

    return V_pi
            


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
    rng = random.Random(seed)

    V_pi = {}

    for episode_id in range(n_episodes):
        episode_seed = rng.randint(0, 2**32 - 1)
        current_state = env.reset(seed=episode_seed)
        step = 0
        
        
        while not env.is_terminal(current_state):
            #TODO: verify if we should take the first action if not found in policy
            time_remaining = current_state[4]
            action = policy.get(time_remaining, {}).get(current_state, env.actions(current_state)[0]) 
            next_state, reward, done, _ = env.step(action)

            if current_state not in V_pi:
                V_pi[current_state] = rng.random()  # Initialize with a random value, we know the state is non-terminal

            alpha_t = alpha_schedule(step)
            next_value = 0.0 if done else V_pi.get(next_state, 0.0)
            td_target = reward + gamma * next_value
            td_error = td_target - V_pi[current_state]
            V_pi[current_state] += alpha_t * td_error

            current_state = next_state
            step += 1
        
        # episode = env.simulate(policy, seed=episode_seed)
        # trajectory = episode["trajectory"]

        # for t, step in enumerate(trajectory):
        #     state = step["state"]
        #     reward = step["reward"]
        #     next_state = step["next_state"]
        #     done = step["done"]

        #     if state not in V_pi:
        #         V_pi[state] = 0.0 if env.is_terminal(state) else rng.random()  # Initialize with a random value for non-terminal states

        #     alpha_t = alpha_schedule(t)
        #     next_value = 0.0 if done else V_pi.get(next_state, 0.0)
        #     td_target = reward + gamma * next_value
        #     td_error = td_target - V_pi[state]
        #     V_pi[state] += alpha_t * td_error

    return V_pi


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
    rng = random.Random(seed)

    Q = {}

    for episode_id in range(n_episodes):
        episode_seed = rng.randint(0, 2**32 - 1)
        current_state = env.reset(seed=episode_seed)
        step = 0

        while not env.is_terminal(current_state):
            epsilon_t = epsilon_schedule(step)
            legal_actions = env.actions(current_state)

            if rng.random() < epsilon_t:
                action = rng.choice(legal_actions)
            else:
                # Choose the greedy action with tie-breaking
                max_q_value = float('-inf')
                best_action = None
                for action in legal_actions:
                    if (current_state, action) not in Q:
                        #TODO: should we initialize with 0 or randomly?
                        Q[(current_state, action)] = 0.0  # Initialize Q-value for unseen state-action pairs
                    q_value = Q[(current_state, action)]           
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_action = action
                action = best_action

            next_state, reward, done, _ = env.step(action)

            # Initialize Q-value for this state-action pair if not seen before
            if (current_state, action) not in Q:
                Q[(current_state, action)] = 0.0

            alpha_t = alpha_schedule(step)
            next_max_q = max(Q.get((next_state, a), 0.0) for a in env.actions(next_state)) if not done else 0.0
            td_target = reward + gamma * next_max_q
            td_error = td_target - Q[(current_state, action)]
            Q[(current_state, action)] += alpha_t * td_error

            current_state = next_state
            step += 1

    # Derive the greedy policy from Q
    #TODO: we can probably optimize this by keeping track of the best action during learning
    policy = {}
    for (state, action), q_value in Q.items():
        if state not in policy or q_value > Q[(state, policy[state])]:
            policy[state] = action

    return Q, policy
            
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
    #TODO: should we assume q value of 0 for unseen state-action pairs? 
    probs = [Q.get((state, action), 0.0) / temperature for action in actions]
    max_q = max(probs)
    exp_probs = [pow(2.718281828459045, q - max_q) for q in probs]
    total = sum(exp_probs)
    normalized_probs = [p / total for p in exp_probs]

    action = random.choices(actions, weights=normalized_probs, k=1)[0]
    return action

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

    rng = random.Random(seed)
    success_count = 0
    total_return = 0.0
    total_length = 0
    failure_counts = {}

    for episode_id in range(n_episodes):
        episode_seed = rng.randint(0, 2**32 - 1)
        episode = env.simulate(policy, seed=episode_seed)
        trajectory = episode["trajectory"]
        total_return += episode["return"]
        total_length += len(trajectory)
        failure_reason = episode["terminal_reason"]

        if failure_reason == 'success':
            success_count += 1
        else:
            
            if failure_reason not in failure_counts:
                failure_counts[failure_reason] = 0
            failure_counts[failure_reason] += 1

    success_rate = success_count / n_episodes
    mean_return = total_return / n_episodes
    mean_length = total_length / n_episodes

    return {
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_length": mean_length,
        "failure_counts": failure_counts,
    }


def default_alpha_schedule(t: int) -> float:
    """Alpha schedule required by the assignment."""

    return 0.5 / (1 + (t // 1000))


def constant_epsilon(_: int) -> float:
    return 0.1


def decaying_epsilon(t: int) -> float:
    return max(0.02, 1 / ((t + 1) ** 0.5))

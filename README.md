# HW3 Rescue Robot Starter Code

Recommended setup with Conda:

```bash
conda env create -f environment.yml
conda activate hw3-rescue
```

If you already created the environment and the file changed, update it with:

```bash
conda env update -f environment.yml --prune
conda activate hw3-rescue
```

Equivalent setup with Micromamba:

```bash
micromamba create -f environment.yml
micromamba activate hw3-rescue
```

On WSL, first move to the project through `/mnt/c`, then use the same Conda or
Micromamba commands:

```bash
cd "/mnt/c/Users/USER001/Documents/HW3 project/starter_code"
conda env create -f environment.yml
conda activate hw3-rescue
```

or:

```bash
cd "/mnt/c/Users/USER001/Documents/HW3 project/starter_code"
micromamba create -f environment.yml
micromamba activate hw3-rescue
```

Run commands from `starter_code/code`.

```bash
python tests_public.py --infra
```

After implementing the required TODO functions, run:

```bash
python tests_public.py
```

To see the completed solution run end-to-end with the assignment-sized smoke
settings, run:

```bash
python run_assignment_smoke.py
```

This prints planning, evaluation, MC/TD, Q-learning, and Boltzmann sanity
outputs, and writes report-style images under `starter_code/smoke_outputs/`.
Before the TODOs are implemented, this script is expected to stop with
`NotImplementedError`.

The public tests are tiny sanity checks. They are not a complete grading suite.
They do, however, check a few required output shapes. In particular,
`estimate_transition_reward_model` must return the dictionary schema documented
in `code/learning_rescue.py`; using that schema will also make
`plan_with_estimated_model` easier to implement.

Main files students edit:

- `planning_rescue.py`
- `learning_rescue.py`

Files students should not edit:

- `rescue_env.py`
- `rescue_types.py`
- `visualize.py`
- files under `data/`

The assignment map is deliberately small. On a normal laptop CPU, finite-horizon
dynamic programming for the requested horizons should finish quickly if your
implementation iterates only over reachable states.

The assignment intentionally uses modest experiment sizes. Larger runs may give
smoother learning curves, but they are not required unless course staff changes
the handout.

Offline rollout files:

- `offline_rollouts_sparse_train.jsonl` and
  `offline_rollouts_sparse_validation.jsonl` are
  sparse, low-quality data. They intentionally contain no successful rescue
  trajectories, so they are useful for analyzing poor data coverage.
- `offline_rollouts_good_train.jsonl` and
  `offline_rollouts_good_validation.jsonl` are good-coverage check files
  collected from the true-model planner. They include successful trajectories
  and show how model-based planning improves when the data contains the success
  signal.

Algorithm map:

- True-model planner, also called the oracle planner: Part B finite-horizon
  dynamic programming. It uses the real `env.transitions(state, action)`, so it
  knows the true transition probabilities and rewards. This is a comparison
  baseline, not a learning-from-data method.
- Transition/reward model estimation: `estimate_transition_reward_model`. This
  learns an estimated model from offline rollouts. It does not directly choose
  actions in the environment.
- Estimated-model planner: `plan_with_estimated_model`. This plans inside the
  learned model and returns a policy, which is then evaluated in the real
  simulator.
- Monte Carlo and TD(0): value-estimation methods for one fixed policy. In this
  assignment they estimate `V^pi`; they are not separate rescue agents in the
  final decision-agent comparison.
- Q-learning: a model-free control agent. It learns `Q(s, a)` from online
  interaction and returns a greedy policy after training.
- Boltzmann exploration: an action-selection rule for exploration from Q-values.
  It is not a separate transition model and not a separate value estimator.

Suggested wet-part workflow:

1. Implement Part B planning first: `action_value`, `finite_horizon_dp`,
   `rollout_time_dependent_policy`, and `compare_policy_slices`.
2. Use the Part B policy as the fixed policy for Monte Carlo and TD(0). These
   two methods estimate values for a policy; they are not decision agents in the
   final agent comparison.
3. For model-based learning, estimate a transition/reward model from
   `offline_rollouts_sparse_train.jsonl`, plan inside that estimated model, then
   evaluate the resulting policy in the real simulator. Repeat the same check
   with `offline_rollouts_good_train.jsonl` to demonstrate the effect of data
   quality.
4. For model-free control, run Q-learning and evaluate the final greedy policy
   without more learning.
5. In the final comparison, compare the oracle planner, the estimated-model
   planner, and Q-learning as decision agents. Discuss Monte Carlo vs TD(0)
   separately as value estimators for the same fixed policy.

Visualization helpers are provided in `code/visualize.py`. Useful entry points:

- `plot_map(env)`
- `plot_policy_slice(env, policy, time_remaining, carrying=False, battery=None)`
- `save_policy_slice(env, policy, output_path, time_remaining, carrying=False, battery=None)`
- `animate_trajectory(env, trajectory, output_path=None)`

To see the visualization helpers before solving the TODOs, run:

```bash
python demo_visualization.py
```

The demo writes example images under `starter_code/demo_outputs/`.

For a fuller explanation of policy slices, rollout animations, and what to say
about the generated images in the report, see `VISUALIZATION_GUIDE.md`.

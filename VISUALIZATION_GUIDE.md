# Visualization Guide

The visualization code is optional helper code for the report. It does not solve
the planning or learning tasks for you, but it makes the policy and rollout
behavior easier to inspect and explain.

Run all commands from `starter_code/code`.

## Quick Demo

Before implementing any TODOs, you can run:

```bash
python demo_visualization.py
```

This writes example files to `starter_code/demo_outputs/`:

- `rescue_map.png`: the rescue grid, including start, pickup, recharge, exit,
  danger, walls, and hazard-colored cell borders.
- `policy_slice_t5_carrying_b2.png`: a policy slice for states with
  `time_remaining=5`, `carrying=True`, and `battery=2`.
- `policy_slice_t15_not_carrying_b1.png`: a second policy slice for
  `time_remaining=15`, `carrying=False`, and `battery=1`.
- `demo_trajectory.gif`: an animated rollout of the demo policy.

The demo uses a simple hand-written policy only to prove the visualization
pipeline works. For your report, use the policy produced by your own planning or
learning code.

## End-To-End Run After Solving

After implementing the TODOs in `planning_rescue.py` and `learning_rescue.py`,
run:

```bash
python run_assignment_smoke.py
```

This script calls your implemented functions with the shortened assignment
budgets. It prints planning and learning metrics, then writes solution-generated
visuals under `starter_code/smoke_outputs/`.

Use these outputs to check that your code is wired together correctly. The exact
numbers can vary with implementation details and random seeds, but the script
should finish without errors and should produce policy-slice images.

## What Is A Policy Slice?

A full policy can contain many states because the state is:

```python
(row, col, carrying_sensor, battery_level, time_remaining)
```

That is too much to show in one picture. A policy slice fixes some of those
state variables and draws only the action chosen at each map location.

For example:

```python
save_policy_slice(
    env,
    policy,
    "my_policy_t5_carrying_b2.png",
    time_remaining=5,
    carrying=True,
    battery=2,
)
```

This image answers:

"If the robot has 5 steps left, is carrying the sensor, and has battery 2, what
action does the policy choose at each grid cell?"

Use at least two meaningful slices in the report. Good choices are slices where
the policy changes because of time pressure, battery level, or whether the robot
is already carrying the sensor.

## Example After Implementing Part B

After implementing `finite_horizon_dp`, you can create policy images like this:

```python
from pathlib import Path

from planning_rescue import finite_horizon_dp
from rescue_env import load_default_env
from visualize import save_policy_slice

env = load_default_env(Path(".."))
V, policy = finite_horizon_dp(env, horizon=20)

output_dir = Path("..") / "my_outputs"
output_dir.mkdir(exist_ok=True)

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
```

## Example Rollout Animation

After you have a policy, you can simulate it and animate the resulting path:

```python
from pathlib import Path

from planning_rescue import finite_horizon_dp
from rescue_env import load_default_env
from visualize import animate_trajectory

env = load_default_env(Path(".."))
V, policy = finite_horizon_dp(env, horizon=20)

result = env.simulate(policy, horizon=20, seed=236501)
animate_trajectory(
    env,
    result["trajectory"],
    Path("..") / "my_outputs" / "true_model_planner_rollout.gif",
)

print(result["return"], result["length"], result["terminal_reason"])
```

If your computer does not have `ffmpeg`, matplotlib may use Pillow to save GIFs.
That is fine.

## Reading The Colors

Cell fill colors:

- `S`: start position.
- `P`: sensor pickup.
- `R`: recharge cell.
- `E`: exit cell. Reaching it succeeds only after picking up the sensor.
- `X`: danger terminal cell.
- Dark cells: walls.

Cell border colors show the hazard type:

- Gray border: normal.
- Dark gray border: smoky.
- Blue border: slippery.
- Orange border: damaged.

Policy markers:

- Arrow: selected movement action.
- `o`: `WAIT`.
- Blank traversable cell: no policy entry exists for that exact state. This
  usually means the state is unreachable in that slice, or that a sparse learned
  model has no learned action for it.
- Empty wall cells have no policy action.

## What To Say In The Report

For each visualization, state the fixed slice values:

- `time_remaining`
- `carrying`
- `battery`

Then explain one concrete behavior visible in the image. For example:

- "With low battery, the policy redirects toward the recharge cell."
- "With little time remaining, the policy takes a riskier route toward the exit."
- "Before pickup, arrows move toward `P`; after pickup, arrows move toward `E`."

The image should support an analysis claim. It should not be included only as
decoration.

# Offline Rollout Data

The assignment includes two kinds of offline data.

Sparse negative-example data:

- `offline_rollouts_sparse_train.jsonl`
- `offline_rollouts_sparse_validation.jsonl`

These files intentionally have poor coverage and no successful rescue
trajectories. They are useful for showing how model-based learning can fail when
the data never contains the success reward.

Good-coverage check data:

- `offline_rollouts_good_train.jsonl`
- `offline_rollouts_good_validation.jsonl`

These files were collected from the true-model planner. They include successful
trajectories and are meant as a sanity check showing that the estimated-model
planner improves when the data contains useful coverage.

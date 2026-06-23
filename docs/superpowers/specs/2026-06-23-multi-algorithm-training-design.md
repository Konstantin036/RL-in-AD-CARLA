# Multi-Algorithm Training Pipeline — Design

Date: 2026-06-23
Status: Approved

## Problem

The training pipeline (`agent/train.py`, `configs/config.yaml`) is hardwired
to PPO. The thesis requires comparing multiple RL algorithms (PPO, SAC,
DDPG, TD3) on the same CARLA lane-keeping task, and the user wants to add
algorithms over time without deleting or rewriting prior work — each
algorithm's code, config, checkpoints, and logs should coexist
side-by-side and be selectable via config/CLI.

`carla_env/env.py` (`CarlaLaneKeepingEnv`) is already a clean Gymnasium
environment with no algorithm-specific logic, so it requires no changes.
The work is entirely in the training entry point and config layer.

## Constraint: DQN is excluded

The action space is continuous (`Box(2,)`: acceleration, steer). DQN in
Stable-Baselines3 only supports discrete action spaces, so it cannot run
against this environment without a separate discretized env variant. Per
user decision, DQN is dropped from scope. **TD3** (DDPG's deterministic
successor, fully continuous) is added in its place. The four supported
algorithms are: **PPO, SAC, DDPG, TD3**.

## Architecture

### 1. `agent/algorithms.py` (new)

Single source of truth for "what algorithms exist and how to build them."

```python
ALGORITHMS = {
    "ppo":  PPO,
    "sac":  SAC,
    "ddpg": DDPG,
    "td3":  TD3,
}
```

Responsibilities:

- `build_model(algo_name, cfg, env, tensorboard_log, resume_path=None)`
  - Looks up the SB3 class from `ALGORITHMS`.
  - If `resume_path` is given: `cls.load(resume_path, env=env, tensorboard_log=tensorboard_log)`.
  - Else: reads the hyperparameter block `cfg[algo_name]` and constructs a
    fresh model. On-policy (PPO) and off-policy (SAC/DDPG/TD3) algorithms
    take different constructor arguments — this function knows the
    difference and only passes what's relevant per algorithm:
    - **PPO** (on-policy): `n_steps`, `n_epochs`, `gae_lambda`, `clip_range`,
      `ent_coef`, `vf_coef`, `max_grad_norm`.
    - **SAC** (off-policy, stochastic): `buffer_size`, `learning_starts`,
      `batch_size`, `tau`, `train_freq`, `gradient_steps`. No action
      noise needed — SAC explores via its stochastic policy.
    - **DDPG / TD3** (off-policy, deterministic): same off-policy
      hyperparameters as SAC, plus `action_noise` (constructed from a
      `action_noise_sigma` config value using
      `NormalActionNoise`), since deterministic policies need explicit
      exploration noise.
  - All algorithms use `MlpPolicy` — the observation space is the same 4D
    low-dimensional vector regardless of algorithm.
- `get_run_prefix(algo_name)` → e.g. `"sac_lane_keeping"`, used for run
  names and checkpoint filename prefixes.

Adding a 5th algorithm later means adding one entry to `ALGORITHMS`, one
branch in the hyperparameter-extraction logic, and one config block — no
changes to `train.py`.

### 2. `configs/config.yaml` changes

- Add a top-level field: `algo: ppo` (default).
- Keep the existing `ppo:` block as-is.
- Add sibling blocks `sac:`, `ddpg:`, `td3:`, each documented inline like
  the current `ppo:` block, containing that algorithm's hyperparameters
  (including `action_noise_sigma` for `ddpg`/`td3`).
- `env:`, `reward:`, `training:`, `paths:` remain shared across all
  algorithms unchanged.

### 3. `agent/train.py` changes

- Add `--algo` CLI argument; if given, overrides `cfg["algo"]`.
- Replace the hardcoded `PPO(...)` / `PPO.load(...)` construction with a
  single call to `algorithms.build_model(...)`.
- Replace the hardcoded `"ppo_lane_keeping"` run name / checkpoint prefix
  with `algorithms.get_run_prefix(cfg["algo"])`.
- Checkpoint, log, and best-model paths become algorithm-scoped:
  `results/checkpoints/{algo}/`, `results/logs/{algo}/`. The `paths:`
  config keys remain base directories; `train()` joins them with the
  algorithm name before use.

### 4. Callbacks — unchanged

`EpisodeLoggerCallback` and SB3's built-in `CheckpointCallback` /
`EvalCallback` operate on the Gymnasium `info` dict and `VecEnv`
interface, neither of which differs by algorithm. No changes needed.

### 5. Environment — unchanged

`CarlaLaneKeepingEnv` requires no modification; this is the existing
payoff of having a clean Gymnasium interface.

### 6. Migration of existing results

Existing PPO checkpoints (`results/checkpoints/*.zip`,
`results/checkpoints/best_model/`) and logs
(`results/logs/ppo_lane_keeping_*`, `results/logs/monitor_*.csv`) move
into `results/checkpoints/ppo/` and `results/logs/ppo/` respectively, so
nothing is lost and the new per-algorithm layout is consistent from the
start. (Note: `results/` is gitignored, so this migration is a filesystem
move, not a git operation.)

### 7. Documentation

Update `CLAUDE.md`:
- Project structure section: add `agent/algorithms.py`, note per-algorithm
  result directories.
- Architecture section: explain the registry pattern and how to add a new
  algorithm.
- "How to run things": show `--algo` usage examples for all four
  algorithms.
- Config section: show the new `algo:` field and per-algorithm blocks.
- Update "Tech stack" / intro language that currently states the agent
  "uses PPO" — reframe as "supports PPO, SAC, DDPG, TD3 via a pluggable
  registry."

## Testing approach

CARLA is running, so both offline and live checks are possible:

1. **Offline construction checks**: for each of the 4 algorithms, call
   `algorithms.build_model()` against a lightweight dummy Gymnasium env
   (matching the real observation/action space shapes) and assert the
   correct SB3 class is returned with no constructor errors. Verifies
   hyperparameter wiring without needing CARLA.
2. **Existing offline tests** (`scripts/test_action.py`,
   `scripts/test_reward.py`) remain untouched and should still pass.
3. **Live smoke tests**: run `python agent/train.py --algo <algo>
   --timesteps <small N>` against the running CARLA instance for each of
   PPO, SAC, DDPG, TD3, confirming each completes without error and
   produces a checkpoint + log directory under the correct per-algorithm
   path.

## Out of scope (for now)

- `agent/evaluate.py` (Phase 9) — building the evaluation/metrics script
  is a separate piece of work, not required for algorithm-switching
  modularity. Can be brainstormed as its own follow-up once this lands.
- Discretized action space / DQN support — not built in this pass, but
  the registry is designed so it can be added later without rework (see
  below).
- Parallel multi-environment training (still 1 CARLA instance, as today).

## Future extension: DQN

DQN is deliberately excluded now (see constraint above) but the user
wants to add it later once a discretized-action variant of the
environment exists. The registry design in this spec keeps that path
open:

- `ALGORITHMS` in `agent/algorithms.py` is just a name → SB3 class map;
  adding `"dqn": DQN` is a one-line change.
- `build_model()`'s per-algorithm hyperparameter branching already
  separates on-policy / off-policy / noise-based logic by `algo_name`,
  so a DQN branch (`exploration_fraction`, `exploration_final_eps`,
  `target_update_interval`, etc.) slots in the same way without touching
  other algorithms.
- The env itself is untouched by this spec. DQN will need a discretized
  action space, which should be a **separate environment variant** (e.g.
  `CarlaLaneKeepingEnvDiscrete` wrapping or subclassing
  `CarlaLaneKeepingEnv`, binning acceleration/steer into N choices) so
  the continuous algorithms (PPO/SAC/DDPG/TD3) are never affected. That
  env variant, plus the DQN registry entry and config block, is its own
  follow-up spec — not part of this implementation.

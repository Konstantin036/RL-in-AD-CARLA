# Model Evaluation Script — Design

## Problem

`agent/train.py` trains and checkpoints models, but there is no way to load
a specific saved checkpoint and measure how well it actually drives outside
of training. The only existing evaluation is SB3's `EvalCallback`, which
runs automatically every `eval_freq` steps *during* training against
whichever model is currently in memory — it cannot be pointed at an
arbitrary saved `.zip`, and it stops existing once training ends.

This was planned from the start as "Phase 9" in `CLAUDE.md` but never built.

## Goal

A standalone script, `agent/evaluate.py`, that loads a saved checkpoint for
any of the four supported algorithms (PPO, SAC, DDPG, TD3) and runs it for N
episodes with its raw, deterministic policy, then reports how well it drove:
mean reward, mean lateral distance, success rate, episode length stats, and
a termination-reason breakdown. Console output plus a CSV for later
analysis. No plots in this version (deferred — see Non-Goals).

## Design

### Operational constraint — run only when no training is live

`CarlaLaneKeepingEnv` connects to the same shared CARLA server `train.py`
uses, and both call `world.tick()` to advance the simulation and
`env.close()` (which disables synchronous mode) on exit. Two clients doing
this against the same world at once causes the world-disruption class of
crash already seen twice this session (see project history — `.close()`
and `load_world()` both crashed live training by touching shared world
state). `evaluate.py` must only be run when no `train.py` process is
connected to the same CARLA server. The script's module docstring and a
startup log line state this explicitly; the script does not attempt to
detect or enforce it programmatically (no reliable way to check this from
the CARLA client API without itself risking the same disruption).

### Modularity

Each responsibility below is a small, independently callable function with
a typed input/output, not one long `main()`. This matters for two reasons:
(1) the existing project convention (`carla_env/reward.py`,
`agent/algorithms.py`) already follows this shape, and (2) it's the
explicit ask for this feature — future additions (plots, multi-checkpoint
comparison, a different success definition) should be addable by writing a
new function that calls `run_evaluation()` or `compute_summary()`, not by
restructuring the script. The CLI (`main()`) is a thin wrapper that wires
these functions together; nothing else depends on argparse.

### 1. `carla_env/reward.py` — extract `RewardConfig.from_dict()`

Both `train.py`'s `make_env()` and the new `evaluate.py` need to turn
`cfg["reward"]` (the YAML dict) into a `RewardConfig`. `train.py` currently
does this inline:

```python
rc = RewardConfig(
    w_center         = reward_cfg["w_center"],
    w_speed          = reward_cfg["w_speed"],
    w_heading        = reward_cfg["w_heading"],
    w_smooth         = reward_cfg["w_smooth"],
    target_speed_kmh = reward_cfg["target_speed_kmh"],
    sigma_speed      = reward_cfg["sigma_speed"],
    terminal_penalty = reward_cfg["terminal_penalty"],
    step_penalty     = reward_cfg["step_penalty"],
)
```

This becomes a classmethod on `RewardConfig`:

```python
@classmethod
def from_dict(cls, reward_cfg: dict) -> "RewardConfig":
    """Build a RewardConfig from configs/config.yaml's `reward:` block."""
    return cls(
        w_center         = reward_cfg["w_center"],
        w_speed          = reward_cfg["w_speed"],
        w_heading        = reward_cfg["w_heading"],
        w_smooth         = reward_cfg["w_smooth"],
        target_speed_kmh = reward_cfg["target_speed_kmh"],
        sigma_speed      = reward_cfg["sigma_speed"],
        terminal_penalty = reward_cfg["terminal_penalty"],
        step_penalty     = reward_cfg["step_penalty"],
    )
```

`train.py`'s `make_env()` is updated to call `RewardConfig.from_dict(reward_cfg)`
instead of the inline construction. This is the only change to existing
files — purely removing duplication, no behavior change.

### 2. `agent/evaluate.py` — new file

**CLI** (mirrors `train.py`'s argument style):

```bash
python agent/evaluate.py --algo sac --checkpoint results/checkpoints/sac/sac_lane_keeping_20260626_111850/best_model/best_model.zip --episodes 20
```

| Flag           | Required | Default                | Meaning                                |
|----------------|----------|-------------------------|-----------------------------------------|
| `--algo`       | yes      | —                       | one of `ppo`, `sac`, `ddpg`, `td3`      |
| `--checkpoint` | yes      | —                       | path to a saved `.zip`                  |
| `--episodes`   | no       | 20                      | number of evaluation episodes to run    |
| `--config`     | no       | `configs/config.yaml`   | same config file `train.py` reads       |

**Data types:**

```python
@dataclass
class EpisodeResult:
    episode_num: int
    reward: float
    length: int
    mean_lateral_distance: float   # mean(|lateral_distance_m|) over the episode's steps
    termination_reason: str        # "timeout" | "collision" | "off_road" | "wrong_heading"

@dataclass
class EvaluationSummary:
    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_lateral_distance: float
    success_rate: float                    # fraction with termination_reason == "timeout"
    mean_length: float
    termination_counts: dict               # {"timeout": 18, "collision": 2, ...}
```

**Functions:**

```python
def build_env(cfg: dict) -> CarlaLaneKeepingEnv:
    """
    Build a single, unwrapped CarlaLaneKeepingEnv from config — no
    Monitor/DummyVecEnv wrapping (those exist for SB3 training internals,
    not needed here). Uses spawn_index_offset=0 (single env, no train/eval
    spawn contention to avoid since nothing else is running).
    """

def load_model(algo_name: str, checkpoint_path: str):
    """Look up the SB3 class via agent.algorithms.ALGORITHMS and call its
    .load(checkpoint_path). Raises ValueError for an unknown algo_name,
    matching build_model()'s existing error style in agent/algorithms.py."""

def run_episode(env: CarlaLaneKeepingEnv, model, episode_num: int) -> EpisodeResult:
    """Run exactly one episode with model.predict(obs, deterministic=True)
    until terminated or truncated. Returns one EpisodeResult."""

def run_evaluation(env, model, n_episodes: int) -> List[EpisodeResult]:
    """Call run_episode() n_episodes times, return the list of results.
    This is the extension point for future work — e.g. a multi-checkpoint
    comparison script can call this once per checkpoint."""

def compute_summary(results: List[EpisodeResult]) -> EvaluationSummary:
    """Pure function: list of EpisodeResult -> EvaluationSummary. No I/O,
    easy to unit test offline without CARLA."""

def write_csv(results: List[EpisodeResult], path: str) -> None:
    """Write one row per EpisodeResult to a CSV at the given path."""

def print_summary(summary: EvaluationSummary) -> None:
    """Print the console report (mean/std reward, success rate, length
    stats, termination-reason table)."""

def main():
    """Parse CLI args, load config, build_env(), load_model(), call
    run_evaluation(), then compute_summary() + write_csv() + print_summary()."""
```

**Output:**
- CSV at `results/logs/{algo}/eval_runs/eval_{checkpoint_stem}_{timestamp}.csv`,
  where `checkpoint_stem` is the checkpoint's filename without directory or
  `.zip` extension (e.g. `--checkpoint .../best_model.zip` →
  `eval_best_model_20260626_143000.csv`), one row per episode
  (`episode_num,reward,length,mean_lateral_distance,termination_reason`).
- Console summary printed after the run: mean ± std reward, mean lateral
  distance, success rate (%), mean episode length, and a termination-reason
  count table.

**Success definition** (confirmed earlier): an episode is successful if it
ends via `truncated=True` (timeout) rather than `terminated=True`
(collision, off-road, or wrong heading) — this is exactly the
`termination_reason == "timeout"` check, reusing `check_termination()`'s
existing distinction from `reward.py` with no new logic.

**Error handling:** `--algo` not in `ALGORITHMS` and a missing
`--checkpoint` file both raise immediately with a clear message, before any
CARLA connection is attempted (fail fast, don't spin up CARLA only to crash
on a typo).

## Non-Goals (deferred, not part of this script)

- **Plots** (matplotlib figures for the thesis) — explicitly deferred to a
  follow-up. `compute_summary()` and `run_evaluation()` are designed so a
  later plotting script can reuse them directly (call `run_evaluation()`,
  get `List[EpisodeResult]`, plot whatever's needed) without touching
  `evaluate.py` itself.
- **Multi-checkpoint comparison** — not built now, but `run_evaluation()`
  taking a single `(env, model, n_episodes)` rather than a list of
  checkpoints means a future comparison script can loop over checkpoints
  and call it once per checkpoint, reusing every function in this file.
- **Automatic detection of a live training process** — not attempted (see
  Operational constraint above); this stays a documented manual precaution.

## Testing

- `compute_summary()` is a pure function (no CARLA, no I/O) — gets offline
  unit tests in `scripts/test_evaluate.py`, following the existing pattern
  of `scripts/test_reward.py` / `scripts/test_algorithms.py` (hand-built
  `EpisodeResult` lists in, `EvaluationSummary` out, assert exact values).
- `load_model()`'s unknown-algo error path is also covered offline (no
  CARLA needed — `ALGORITHMS` lookup fails before any env is touched).
- `build_env()`, `run_episode()`, and `main()` require a live CARLA server
  and are verified manually (run the script against a real checkpoint with
  CARLA running, training stopped) rather than via automated tests — same
  approach already used for `carla_env/env.py`'s integration behavior.

# CLAUDE.md — Project Context for Claude Code

This file gives Claude Code full context about the project so it can
help without needing re-explanation. Read this before touching any file.

---

## What this project is

A reinforcement learning system that trains a car to keep its lane
in the CARLA driving simulator. The agent uses PPO (Proximal Policy
Optimization) from Stable-Baselines3. The observation is a 4D
low-dimensional state vector. The action is 2D continuous control.

This is a **graduate thesis project**. Code must be:
- Clean and readable (will be explained in a thesis defense)
- Modular (each file has one clear responsibility)
- Well commented (comments explain WHY, not just WHAT)

---

## Tech stack

| Component        | Library / Tool              | Version     |
|------------------|-----------------------------|-------------|
| Simulator        | CARLA                       | 0.9.15      |
| RL algorithm     | Stable-Baselines3 PPO       | 2.0.0       |
| Gym interface    | Gymnasium                   | 0.28.1      |
| Neural network   | PyTorch                     | 1.13.1      |
| Python           | CPython                     | 3.7.16      |
| OS               | Ubuntu 22.04                |             |
| Conda env        | carla915                    |             |

---

## Project structure

```
carla_rl_project/
│
├── carla_env/                  # The RL environment — core thesis contribution
│   ├── __init__.py
│   ├── env.py                  # Main Gym environment (CarlaLaneKeepingEnv)
│   ├── observation.py          # Builds the 4D observation vector
│   ├── action.py               # Action space + smoother + CARLA control mapping
│   ├── reward.py               # Reward function + termination conditions
│   └── sensors.py              # Collision sensor wrapper
│
├── agent/                      # RL training pipeline
│   ├── __init__.py
│   ├── train.py                # PPO training entry point
│   ├── evaluate.py             # Evaluation script (Phase 9 — not yet built)
│   └── callbacks.py            # SB3 callbacks: episode logger, checkpoints
│
├── scripts/                    # Utility and testing scripts
│   ├── verify_carla.py         # Phase 1: confirm CARLA connects
│   ├── spawn_test.py           # Phase 2: spawn vehicle and tick
│   ├── test_observation.py     # Phase 3: live observation values
│   ├── test_action.py          # Phase 4: offline action unit tests
│   ├── test_reward.py          # Phase 5: offline reward unit tests
│   ├── test_env.py             # Phase 6: full env integration test
│   └── manual_drive.py         # Phase 7: keyboard manual control
│
├── configs/
│   └── config.yaml             # ALL hyperparameters — single source of truth
│
├── results/                    # Generated at runtime — not committed to git
│   ├── logs/                   # TensorBoard logs + episode CSV
│   ├── checkpoints/            # Saved model weights
│   └── plots/                  # Evaluation plots (Phase 9)
│
├── docs/
│   └── architecture.md         # Thesis-supporting architecture notes
│
├── CLAUDE.md                   # ← you are here
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Architecture overview

### How the RL loop works

```
PPO (train.py)
    │
    ▼
CarlaLaneKeepingEnv (env.py)
    │
    ├── reset()
    │     ├── _destroy_actors()        destroy previous vehicle + sensor
    │     ├── world.tick() × 2         let CARLA process destructions
    │     ├── _spawn_vehicle()         spawn Tesla Model 3 at fixed/random point
    │     ├── CollisionSensor(...)     attach collision sensor
    │     ├── world.tick() × SETTLE   let physics stabilize
    │     └── compute_observation()   return first obs
    │
    └── step(action)
          ├── ActionProcessor.process(action)   smooth + apply VehicleControl
          ├── world.tick()                       advance simulation 0.05s
          ├── compute_observation()              build 4D obs vector
          ├── check_termination()                collision / off_road / timeout
          ├── compute_reward()                   dense reward signal
          └── return (obs, reward, terminated, truncated, info)
```

### Observation vector (4D)

```
Index  Name              Raw range         Normalized   Source
  0    lateral_distance  -3.5 … +3.5 m    -1 … +1      waypoint projection
  1    heading_error     -π … +π rad       -1 … +1      yaw difference
  2    speed             0 … 80 km/h       0 … +1       velocity magnitude
  3    steering          -1 … +1           -1 … +1      vehicle.get_control()
```

### Action vector (2D)

```
Index  Name          Range     Mapping
  0    acceleration  -1 … +1   >0 → throttle,  <0 → brake
  1    steer         -1 … +1   directly to CARLA steer
```

Action smoothing: `smoothed = 0.6 * new + 0.4 * previous`
Prevents jerky oscillations from Gaussian policy sampling.

### Reward function

```
r = w_center  * (1 - |lat| / max_lat)
  + w_speed   * exp(-((spd - target)² / (2σ²)))
  + w_heading * (1 - |hdg| / π)
  + terminal_penalty   (only on collision/off_road)
  + step_penalty       (every step)
```

Current weights (in configs/config.yaml):
- w_center = 1.0, w_speed = 1.5, w_heading = 0.5
- target_speed = 30 km/h, sigma = 10 km/h
- terminal_penalty = -10.0, step_penalty = -0.1

### Termination conditions

```
terminated (agent's fault — apply terminal penalty):
    - collision sensor fired
    - |lateral_distance| >= 3.5 m  (off road)
    - |heading_error| >= 90°       (pointing wrong way)

truncated (timeout — no penalty):
    - step_count >= max_steps (1000 steps = 50 simulated seconds)
```

---

## CARLA setup

### Launch CARLA server

```bash
cd /path/to/carla
./CarlaUE4.sh -quality-level=Low -fps=20
```

### Activate conda environment

```bash
conda activate carla915
```

### CARLA Python API

The CARLA egg is on PYTHONPATH via the conda environment.
Python 3.7.16 is required for the CARLA 0.9.15 egg.

---

## How to run things

```bash
# Verify CARLA connection
python scripts/verify_carla.py

# Run offline unit tests (no CARLA needed)
python scripts/test_action.py
python scripts/test_reward.py

# Run live integration test (CARLA must be running)
python scripts/test_env.py

# Manual driving (CARLA must be running)
python scripts/manual_drive.py

# Train PPO agent (CARLA must be running)
python agent/train.py
python agent/train.py --timesteps 10000    # quick test
python agent/train.py --resume results/checkpoints/best_model

# Monitor training
tensorboard --logdir results/logs
```

---

## Current training config (configs/config.yaml)

```yaml
env:
  map_name:     Town04      # highway loop, no intersections
  max_steps:    1000        # 50 simulated seconds per episode
  spawn_index:  0           # fixed spawn point for reproducibility

reward:
  w_speed:      1.5         # high weight to prevent standing still
  step_penalty: -0.1        # encourages efficient driving

ppo:
  learning_rate: 0.0003
  n_steps:       2048
  batch_size:    64
  gamma:         0.99

training:
  total_timesteps: 500000
```

---

## Known issues and design decisions

### Why synchronous mode?
CARLA's async mode runs at variable FPS. Synchronous mode with
fixed_delta_seconds=0.05 gives exactly 20 physics steps per second,
making training reproducible. Without this, two runs with the same
seed give different results.

### Why action smoothing?
PPO's Gaussian policy can output large action changes between steps.
Without smoothing, steering oscillates badly and the vehicle swerves.
Alpha=0.6 gives fast enough response while preventing oscillation.

### Why separate terminated vs truncated?
Gymnasium convention. `terminated=True` means the agent failed —
apply the terminal penalty and bootstrap from zero. `truncated=True`
means timeout — bootstrap from the value function normally. Mixing
them causes PPO to fear timeouts as much as crashes.

### Why Town04?
Town04 has a highway loop with no intersections. Good for validating
basic lane keeping before moving to complex maps. Town03 has
intersections and tight curves — use that in later training phases.

### Why fixed spawn point during early training?
Easier to visually verify improvement — the car always starts at
the same position so you can see how far it gets before failing.
Switch to random spawns after basic lane keeping is learned.

### Ghost vehicle bug (fixed)
Fast episode resets caused two vehicles to exist simultaneously.
Fix: tick world twice after destroying actors before spawning new one.
Also: filter and destroy any stray Tesla actors in _destroy_actors().

---

## What has NOT been built yet

These phases are planned but not implemented:

- **Phase 9** — `agent/evaluate.py`: run a trained model for N episodes,
  compute metrics (mean reward, mean lateral distance, % successful
  episodes), generate matplotlib plots for thesis figures.

- **Phase 10** — `docs/architecture.md`: full thesis documentation
  of system design, reward function derivation, training curves,
  and experimental results.

- **Camera/lidar sensors** — `carla_env/sensors.py` currently only
  has CollisionSensor. CameraSensor and LidarSensor can be added
  for a high-dimensional observation extension.

- **Baseline controller** — a PID or pure pursuit controller for
  comparison against the RL agent. Goes in `agent/baseline.py`.

---

## Coding rules for this project

1. Python 3.7 compatible syntax only
2. No type hints that require Python 3.9+ (no `list[int]`, use `List[int]`)
3. All CARLA imports inside functions (not at module top level)
   — allows offline testing without CARLA installed
4. All gymnasium imports lazy where possible for same reason
5. Always use try/finally for CARLA cleanup
6. Always destroy sensors before vehicles
7. Always disable sync mode in close() and finally blocks
8. Use logger (not print) inside carla_env/ and agent/
9. Use print inside scripts/ (simpler, no logging setup needed)
10. Keep reward weights in config.yaml — never hardcode them

---

## Thesis context

- Student: Konstantin (rtrk)
- Topic: Reinforcement learning for autonomous vehicle lane keeping
- Simulator: CARLA 0.9.15
- Algorithm: PPO with MlpPolicy (4D obs → 64×64 → 2D action)
- Task: Lane keeping on Town04 highway
- Evaluation metrics: mean episode reward, mean lateral distance,
  episode length distribution, termination reason breakdown

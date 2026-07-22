# CARLA Reinforcement Learning — Lane Keeping Agent

Thesis project comparing four reinforcement learning algorithms for autonomous vehicle **lane keeping** in the CARLA 0.9.15 driving simulator.

**Algorithms compared:** PPO · SAC · DDPG · TD3  
**Framework:** Stable-Baselines3 2.0.0 · Gymnasium 0.28.1 · Python 3.7.16  
**Simulator:** CARLA 0.9.15 · Town04 highway loop · Synchronous mode

---

## Results

Both trained algorithms achieve **100% success rate** (full 50-second episodes, no crashes or off-road exits) under deterministic evaluation with identical protocol — 20 episodes, fixed spawn point.

| Algorithm | Training Steps | Mean Reward | Lateral Distance | Success Rate |
|-----------|---------------|-------------|-----------------|--------------|
| **PPO** | ~1M | 3280.89 ± 0.09 | 0.024 m | 100% |
| **SAC** | ~1M | 3325.06 ± 0.09 | **0.015 m** | 100% |
| DDPG | — | — | — | not yet trained |
| TD3 | — | — | — | not yet trained |

---

## Figures

### Training Curves

Episode reward over the full training history for each algorithm. Raw episodes (faded) and 20-episode rolling mean (solid). PPO shows steady improvement from ~1000 to ~3100 reward over 1M steps. SAC converges faster but with higher variance early in training (off-policy replay buffer warmup).

![Training Curves](docs/figures/training_curves.png)

---

### Algorithm Performance Comparison

Bar chart comparing final deterministic evaluation metrics: mean episode reward (±std), success rate, and mean lateral distance from lane centre. Lower lateral distance = better lane centering. Both algorithms reach 100% success; SAC achieves tighter centering (0.015 m vs 0.024 m).

![Performance Comparison](docs/figures/comparison_bars.png)

---

### Lane Centering Progress

Mean lateral distance from lane centre per episode throughout training. Both algorithms achieve sub-0.5 m centering (dashed line) early in training and approach the 0.1 m "excellent" threshold (dotted line) by the end. SAC reaches finer centering due to automatic entropy tuning.

![Lane Centering Progress](docs/figures/lateral_progress.png)

---

### Episode Termination Breakdown

Distribution of episode termination reasons in deterministic evaluation. 100% of episodes reach the maximum step count (timeout = success) for both PPO and SAC — no crashes, no off-road exits, no wrong-heading terminations.

![Termination Breakdown](docs/figures/termination_breakdown.png)

---

## System Architecture

### Observation Space (4D)

| Index | Feature | Raw Range | Normalised |
|-------|---------|-----------|------------|
| 0 | lateral distance from lane centre | −3.5 … +3.5 m | −1 … +1 |
| 1 | heading error | −π … +π rad | −1 … +1 |
| 2 | vehicle speed | 0 … 80 km/h | 0 … +1 |
| 3 | current steering | −1 … +1 | −1 … +1 |

### Action Space (2D continuous)

| Index | Action | Range | Effect |
|-------|--------|-------|--------|
| 0 | acceleration | −1 … +1 | > 0 → throttle, < 0 → brake |
| 1 | steering | −1 … +1 | direct CARLA steer |

Action smoothing: `smoothed = 0.6 × new + 0.4 × previous` — prevents oscillation from stochastic policy sampling.

### Reward Function

```
r = 1.0 × (1 − |lateral| / 3.5)        # lane centering
  + 1.5 × exp(−((speed − 30)² / 200))   # speed target 30 km/h
  + 0.5 × (1 − |heading| / π)           # heading alignment
  + 0.5 × smoothness                     # penalise abrupt steering
  − 0.1                                  # per-step cost
  − 10.0  (on collision or off-road)     # terminal penalty
```

### Termination Conditions

| Condition | Type |
|-----------|------|
| `\|lateral\| ≥ 3.5 m` | terminated (off road) |
| Collision sensor fired | terminated |
| `\|heading\| ≥ 90°` | terminated |
| 1000 steps reached | truncated (success) |

---

## Project Structure

```
carla_rl_project/
├── carla_env/              # Gym environment
│   ├── env.py              # CarlaLaneKeepingEnv
│   ├── observation.py      # 4D state vector
│   ├── action.py           # action smoother + CARLA control mapping
│   ├── reward.py           # reward function + termination
│   └── sensors.py          # collision sensor
├── agent/
│   ├── algorithms.py       # PPO / SAC / DDPG / TD3 registry
│   ├── train.py            # multi-algorithm training entry point
│   ├── evaluate.py         # deterministic evaluation script
│   └── callbacks.py        # episode logger + checkpoint callbacks
├── scripts/
│   ├── generate_metrics.py # print comparison table from CSV data
│   └── plot_metrics.py     # generate the four thesis figures
├── configs/
│   └── config.yaml         # all hyperparameters — single source of truth
├── docs/
│   ├── figures/            # thesis figures (committed, visible on GitHub)
│   └── progress_report.md  # full progress report for thesis supervisor
└── results/                # generated at runtime — not committed
    ├── logs/{algo}/        # episode CSVs + TensorBoard logs
    └── checkpoints/{algo}/ # model weights
```

---

## Setup

### Requirements

- Ubuntu 22.04
- CARLA 0.9.15
- Conda (Miniconda or Anaconda)
- GPU with ≥ 6 GB VRAM

### Environment

```bash
conda create -n carla915 python=3.7.16
conda activate carla915
pip install -r requirements.txt
```

The CARLA Python egg is added to the `carla915` conda environment's `sitecustomize.py` — no manual PYTHONPATH needed after setup.

### Launch CARLA

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh -quality-level=Low -nosound
```

---

## Usage

```bash
conda activate carla915
cd carla_rl_project

# Train an algorithm (CARLA must be running)
python agent/train.py --algo ppo
python agent/train.py --algo sac
python agent/train.py --algo ddpg
python agent/train.py --algo td3

# Resume from checkpoint
python agent/train.py --algo ppo --resume results/checkpoints/ppo/.../final_model.zip

# Evaluate a trained model (CARLA must be running)
python agent/evaluate.py --algo ppo \
    --checkpoint results/checkpoints/ppo/.../best_model/best_model.zip \
    --episodes 20

# Generate metrics table (no CARLA needed)
python scripts/generate_metrics.py

# Generate all figures (no CARLA needed)
python scripts/plot_metrics.py

# Monitor training
tensorboard --logdir results/logs
```

---

## Key Findings

1. **Both on-policy (PPO) and off-policy (SAC) RL solve lane keeping** with a 4D observation — no camera or lidar needed for a straight highway segment.
2. **SAC achieves better lane centering** (0.015 m vs 0.024 m) due to automatic entropy tuning, which avoids the stand-still local optimum that PPO is vulnerable to.
3. **PPO requires careful entropy tuning** — `ent_coef=0.05` was necessary after the agent collapsed to a stationary policy at `ent_coef=0.01`.
4. **PPO converged at ~681k steps** — additional training to 1M steps did not improve the best checkpoint, suggesting early convergence.
5. **Action smoothing (α=0.6) is critical** — without it, PPO's stochastic policy produces steering oscillations that destabilise the vehicle.

---

## References

- [CARLA Simulator](https://carla.org)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
- [Gymnasium](https://gymnasium.farama.org)
- [PPO — Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
- [SAC — Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)
- [TD3 — Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)
- [DDPG — Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971)

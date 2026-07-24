# CARLA Reinforcement Learning — Lane Keeping Agent

Thesis project comparing reinforcement learning algorithms for autonomous vehicle lane keeping in CARLA 0.9.15. The car observes four numbers — lateral displacement from lane center, heading error, speed, and its own last steering command — and learns to output throttle/brake and steering to stay centered at 30 km/h.

**Algorithms:** PPO · SAC · DDPG · TD3  
**Framework:** Stable-Baselines3 2.0.0 · Gymnasium 0.28.1 · Python 3.7.16  
**Simulator:** CARLA 0.9.15 · Town04 highway loop · Synchronous mode (20 Hz)

---

## Results

Each algorithm runs 20 evaluation episodes with a fixed spawn point and no action noise. An episode ends either at 1000 steps — 50 simulated seconds at ~30 km/h — or earlier if the car crashes, leaves the lane, or points the wrong way. Reaching 1000 steps is a success. All four algorithms were evaluated with the same reward function so episode rewards are directly comparable.

| Algorithm | Training Steps | Episode Reward (mean ± std, 20 episodes) | Mean Lateral Distance | Success Rate |
|-----------|---------------|------------------------------------------|-----------------------|--------------|
| **PPO** | ~1M | 3280.89 ± 0.09 | 0.024 m | 100% |
| **SAC** | ~1M | **3325.06 ± 0.09** | **0.015 m** | 100% |
| **TD3** | 500k | 3129.90 ± 0.30 | 0.448 m | 100% |
| **DDPG** | 500k | 1392.37 ± 6.94 | 2.108 m | 100% |

**Episode reward** is the sum of per-step rewards over 1000 steps (see Reward Function). A perfect run — centered at exactly 30 km/h, zero heading error, perfectly smooth steering — scores roughly 3400. All four algorithms achieve 100% success: every evaluation episode runs the full 1000 steps without collision or lane departure. The difference is centering precision. SAC stays within 1.5 cm of lane center on average; PPO within 2.4 cm; TD3 within 44.8 cm; DDPG at 210.8 cm — effectively driving along the lane edge for the entire episode. The near-zero standard deviations for PPO and SAC confirm their deterministic evaluation policies trace nearly identical paths every run. TD3 and DDPG show slightly higher variance, consistent with their coarser control.

---

## Figures

### Training Curves

The y-axis is total episode reward — the sum of per-step shaped rewards. The x-axis counts how many environment steps the policy has processed. Faded points are individual training episodes; the solid line is a 20-episode rolling mean. PPO climbs steadily from ~1000 to ~3100 over 1M steps — on-policy updates process each batch of experience once. SAC starts low during the first ~10k steps while the replay buffer fills with random-action data, then improves quickly. DDPG and TD3 each spent 50k steps in pure random exploration before the learned policy activated, then converged rapidly once the Q-function had enough driving experience. Their training rewards appear inflated relative to PPO and SAC because training used an additional forward-progress reward component (see Exploration section); evaluation rewards are directly comparable across all algorithms.

![Training Curves](docs/figures/training_curves.png)

---

### Algorithm Performance Comparison

Bar heights are means over 20 deterministic evaluation episodes; error bars show ±1 standard deviation. For episode reward, higher means the car stayed centered, drove at target speed, and maintained good heading for longer. For lateral distance, lower means the car stayed closer to lane center. PPO and SAC have near-zero error bars — deterministic evaluation at a fixed spawn traces almost the same path every run. TD3 and DDPG show wider bars and larger error: their policies converged to driving but with less precision, so small physical differences between episodes produce more reward variation.

![Performance Comparison](docs/figures/comparison_bars.png)

---

### Lane Centering Progress

The y-axis is mean lateral distance per training episode — average displacement from lane center during that episode. The 0.5 m dashed line marks the point where the car's drift is visible to an observer; the 0.1 m dotted line marks tight centering. PPO and SAC reach and hold sub-0.1 m during the converged phase, consistent with their evaluation results. TD3 stabilizes around 0.2–0.5 m and DDPG around 2.0 m — the deterministic policies learned to drive but without the centering precision that entropy regularization produces. The gap between entropy-based algorithms (top two plots) and deterministic algorithms (bottom two) is the main visual finding of this figure.

![Lane Centering Progress](docs/figures/lateral_progress.png)

---

### Episode Termination Breakdown

Each evaluation episode ends for one of four reasons: lane exit, collision, wrong heading (>90° from road direction), or reaching the 1000-step limit. Reaching the limit is a success — the car drove 50 seconds without any failure. All four algorithms terminate 100% of episodes at the step limit. There are no collisions, lane departures, or wrong-heading events in any of the 80 total evaluation episodes (20 per algorithm). Survival is necessary but not sufficient: DDPG survives while driving 2.1 m off-center, which would be a lane departure in a real two-lane road. The lateral distance metric distinguishes policy quality where the binary success/failure measure cannot.

![Termination Breakdown](docs/figures/termination_breakdown.png)

---

### Sample Efficiency

All training curves on a single time axis. The dashed line at reward = 2500 marks a policy that keeps the car in the lane reliably — below this threshold, the car still fails too often to be useful. Vertical markers show when each algorithm first crossed this threshold. An earlier crossing means fewer environment interactions were needed to reach useful behavior. This matters because each CARLA step takes real wall-clock time; sample efficiency directly determines how long training takes.

![Sample Efficiency](docs/figures/sample_efficiency.png)

---

### Multi-Metric Radar Chart

Six axes represent six dimensions of performance. Each axis is normalized to a realistic range for this specific task — not to the theoretical maximum — so an algorithm that performs well appears as a large polygon, and differences between two good algorithms remain visible. The axes:

- **Reward quality** — episode reward mapped to [2500, 3500]. Below 2500 the policy is unreliable; 3500 is near-perfect.
- **Lane centering** — inverted lateral distance mapped to [0.85 m, 1.0 m] offset. The outer edge corresponds to 15 cm average displacement; the center of the axis is perfect centering.
- **Speed adherence** — how much of the episode the car spent near the 30 km/h target.
- **Steering smoothness** — how little the steering command changed between consecutive steps.
- **Success rate** — fraction of episodes completing without any failure.
- **Sample efficiency** — how quickly the policy crossed the 2500-reward threshold, mapped to [0, 1M] steps.

![Radar Chart](docs/figures/radar_chart.png)

---

### Evaluation Score Distribution

Box plots of episode reward and mean lateral distance across the 20 evaluation episodes. The box spans the middle 50% of episodes (interquartile range); the line inside is the median; individual episode dots are overlaid with a small horizontal jitter. PPO and SAC show near-zero spread (reward std ≈ 0.09) because their deterministic evaluation policies trace nearly the same path every run. TD3 shows moderate spread (std ≈ 0.30), and DDPG the most (std ≈ 6.94), consistent with less precise control. The lateral distance subplot reveals the centering gap more sharply: SAC and PPO boxes cluster near zero, while TD3 and DDPG boxes sit at 0.45 m and 2.1 m — differences invisible in the termination breakdown but critical for real-world lane keeping.

![Evaluation Distributions](docs/figures/eval_distributions.png)

---

### Training Stability

The y-axis is the rolling standard deviation of episode reward over a 50-episode window — how much performance varied within each window. PPO drops to near-zero variance by ~200k steps and stays there. SAC shows periodic spikes, each corresponding to a training session restart where the replay buffer starts empty and performance dips until the buffer fills. TD3 and DDPG show a characteristic two-phase shape: very high variance during the 50k-step random exploration phase (episodes alternate between stalling and driving, which are very different reward magnitudes), then a sharp drop once the learned policy stabilizes. DDPG's converged variance is higher than TD3's, consistent with its coarser policy.

![Training Stability](docs/figures/training_stability.png)

---

### Speed Distribution

Histogram of mean episode speed from the last 300 training episodes of each algorithm's final run, covering the converged phase. Episodes with mean speed below 20 km/h are excluded — they come from random exploration before the learned policy activates. The dashed line at 30 km/h is the reward function's target. SAC clusters tightly around 30 km/h; PPO peaks near 28.5 km/h with a wider spread — its stochastic policy samples different throttle values each step. TD3 and DDPG drive above target speed (frequently exceeding 30 km/h) because the forward-progress reward used during their training rewards movement proportional to speed up to target, creating slight incentive to stay at or above the target rather than match it precisely.

![Speed Distribution](docs/figures/speed_distribution.png)

---

## System Architecture

### Observation Space

The agent receives four numbers per step. All are normalized before reaching the network so the network sees values in a consistent range regardless of physical units.

| Index | What it measures | Physical units | Normalized range | Why it is included |
|-------|-----------------|----------------|-----------------|-------------------|
| 0 | Lateral distance from lane center | −3.5 … +3.5 m | −1 … +1 | Primary task signal — what the car must minimize |
| 1 | Heading error (car vs road direction) | −π … +π rad | −1 … +1 | Tells the car how much to steer to realign |
| 2 | Speed | 0 … 80 km/h | 0 … +1 | Needed to hit the 30 km/h target |
| 3 | Previous steering command | −1 … +1 | −1 … +1 | Lets the network account for its own momentum |

Index 3 (previous steering) matters because the car's steering response has inertia. Seeing its own last command lets the network learn that large step-to-step changes cause oscillation and steer more smoothly.

### Action Space

The network outputs two numbers per step:

| Index | What it controls | Range | How it maps to CARLA |
|-------|-----------------|-------|----------------------|
| 0 | Longitudinal control | −1 … +1 | Positive → throttle, negative → brake |
| 1 | Steering | −1 … +1 | Sent directly to CARLA's steer input |

Actions are smoothed before reaching CARLA: `smoothed = 0.6 × new + 0.4 × previous`. Without this, a stochastic policy's sample-to-sample variation produces steering changes faster than the car can physically follow, causing the trajectory to oscillate.

### Reward Function

At each of the 1000 steps in an episode, the agent receives a scalar reward composed of four terms:

```
r = 1.0 × (1 − |lateral| / 3.5)        # 0 at lane edge, 1 when centered
  + 1.5 × exp(−((speed − 30)² / 200))   # Gaussian peak at 30 km/h, σ = 10 km/h
  + 0.5 × (1 − |heading| / π)           # 0 pointing sideways, 1 pointing along road
  + 0.5 × smoothness                     # 0 for large steering change, 1 for no change
  − 0.1                                  # constant per-step cost
  − 10.0  (only on collision or off-road) # terminal penalty, applied once
```

The speed term uses a Gaussian so the car gets partial credit for being near 30 km/h. At 25 km/h the car receives 78% of the peak speed reward; at 0 km/h it receives nearly nothing. Without this, a stationary car can score reasonable rewards from centering and smoothness alone — the stand-still local optimum. The per-step cost (−0.1) discourages unnecessary stopping and rewards efficient motion.

### Termination Conditions

| Trigger | Gymnasium signal | Consequence for learning |
|---------|-----------------|--------------------------|
| 1000 steps elapsed | `truncated` | Value function bootstraps from the estimated value at the final state |
| `|lateral| ≥ 3.5 m` | `terminated` | Terminal penalty applied; value bootstrap from zero |
| Collision sensor fires | `terminated` | Terminal penalty applied; value bootstrap from zero |
| `|heading| ≥ 90°` | `terminated` | Terminal penalty applied; value bootstrap from zero |

The `terminated`/`truncated` distinction matters for learning. A truncated episode hit a time limit — the car was doing fine and ran out of allowed steps. Treating it as a failure would make the policy avoid long successful runs. With the correct signal, the value function can estimate what reward the car would have continued to collect past the cutoff.

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
│   └── plot_metrics.py     # generate all thesis figures
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

## Exploration vs Exploitation

The four algorithms differ fundamentally in how they balance exploring new actions against repeating actions that already work.

| Algorithm | Type | Policy | How it explores | Entropy control |
|-----------|------|--------|----------------|----------------|
| **PPO** | On-policy | Stochastic | Entropy bonus (`ent_coef=0.05`) penalizes policies that concentrate on one action | Fixed — set before training |
| **SAC** | Off-policy | Stochastic | Same entropy mechanism, but the target entropy level is learned automatically | **Automatic** — adjusts throughout training |
| **TD3** | Off-policy | Deterministic | Ornstein-Uhlenbeck noise (σ=0.3, θ=0.15) added to actions; temporally correlated so consecutive steps push in the same direction | None — noise schedule is fixed |
| **DDPG** | Off-policy | Deterministic | Same OU noise as TD3; TD3 additionally uses target policy smoothing and clipped double Q-functions to reduce overestimation | None — noise schedule is fixed |

In practice:

- PPO must have its entropy coefficient set correctly before training. `ent_coef=0.01` caused the policy to lock onto the stand-still local optimum — a stationary car earns ~1880 reward per episode from centering and smoothness. `ent_coef=0.05` provides enough exploration pressure to escape it.
- SAC learns the right exploration level automatically and avoids the stand-still trap without manual tuning.
- DDPG and TD3 initially used independent Gaussian noise, which averaged to zero through the action smoother (α=0.6) and could not push the car consistently. Switching to Ornstein-Uhlenbeck noise (temporally correlated — consecutive steps stay positive or negative for sustained bursts) was necessary but not sufficient. Both algorithms also required a forward-progress reward component (`w_progress × speed/target`) that is strictly zero at zero speed; without it, centering and smoothness rewards gave the stand-still state a non-zero baseline the Q-function could exploit. With both changes and 50k random-exploration steps before policy activation, both algorithms eventually drove — but converged to coarser policies than the entropy-based methods.
- SAC, DDPG, and TD3 store all past experience in a replay buffer and sample from it repeatedly. PPO discards each batch after one update pass. This is why off-policy methods typically reach useful behavior with fewer environment interactions.

---

## Key Findings

1. **A 4D observation is sufficient** for lane keeping on a straight highway. The car does not need cameras or lidar — it only needs to know where it is laterally, which way it is pointing, how fast it is going, and what steering it applied last step.

2. **Entropy regularization determines centering precision, not just survival.** All four algorithms achieve 100% episode survival. The centering gap is: SAC 0.015 m, PPO 0.024 m, TD3 0.448 m, DDPG 2.108 m. SAC and PPO — both entropy-based — center 10–140× more precisely than the deterministic algorithms. Survival and centering are different skills; the binary success metric is insufficient to distinguish algorithm quality here.

3. **Deterministic policies collapse to a stand-still local optimum without intervention.** DDPG and TD3 both initially found that standing still earns ~1880 reward per episode from centering and smoothness — competitive with early driving attempts. Two interventions were required to escape it: (a) Ornstein-Uhlenbeck exploration noise instead of independent Gaussian noise (which averages to zero through action smoothing and cannot sustain directional pressure), and (b) a forward-progress reward component (`w_progress × min(speed, target)/target`) that is exactly zero at zero speed, removing the centering baseline. PPO and SAC escaped the same optimum through entropy alone.

4. **SAC centers more tightly than PPO** (0.015 m vs 0.024 m) because automatic entropy tuning finds the right exploration level without manual intervention. It neither overexplores (erratic steering) nor underexplores (stand-still).

5. **PPO is sensitive to entropy tuning**. At `ent_coef=0.01` the policy collapsed to standing still within the first ~100k steps. `ent_coef=0.05` provides enough exploration pressure to escape it. This makes PPO more fragile to tune than SAC.

6. **PPO's best checkpoint appeared at ~681k steps**. Continuing to 1M steps slightly degraded performance — the policy had already converged. Evaluation uses the 681k checkpoint.

7. **Action smoothing (α=0.6) is required for stochastic policies**. Without smoothing, the step-to-step steering variance in PPO's Gaussian policy causes oscillation the car cannot physically follow.

---

## References

- [CARLA Simulator](https://carla.org)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
- [Gymnasium](https://gymnasium.farama.org)
- [PPO — Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
- [SAC — Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)
- [TD3 — Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)
- [DDPG — Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971)

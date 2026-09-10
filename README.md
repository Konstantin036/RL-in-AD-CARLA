# Reinforcement Learning for Autonomous Driving in CARLA

**Lane keeping, route following, and traffic-light compliance at signalized intersections — a comparative study of PPO, SAC, DDPG, and TD3.**

This repository holds the practical implementation behind a graduate diploma thesis on reinforcement learning for autonomous vehicle control. A simulated vehicle learns, from a low-dimensional state vector and no hand-coded driving rules beyond a reward signal, to follow an arbitrary planned route through an urban road network, stay centered in its lane, and stop for red lights — evaluated head-to-head across four RL algorithms trained under an identical, equal-budget setup.

---

## 1. Problem and scope

Lane keeping alone — staying centered on a straight or gently curving road — is a well-studied RL benchmark. This project extends it in two directions that are considerably harder to get right: **navigating through real signalized intersections along a planned route**, and **complying with traffic-light state** as a first-class part of the task, not an afterthought. Both require resolving an ambiguity that a simple "stay near the road" objective cannot: at a junction, several lanes belonging to crossing or turning paths are all physically close to the vehicle, and only the route actually intended for that episode disambiguates which one matters.

The simulated agent operates in [CARLA](https://carla.org) 0.9.15's `Town10HD_Opt` map, a mixed urban environment with real intersections, traffic signals, and multi-lane roads.

## 2. System architecture

### 2.1 Observation and action spaces

| Observation (5D, normalized to ≈[-1, 1]) | Action (2D, continuous) |
|---|---|
| Lateral distance from the planned route's centerline | Acceleration (throttle/brake) |
| Heading error relative to the route direction | Steering |
| Vehicle speed | |
| Previous steering command | |
| Traffic-light state (must-stop / clear) | |

The route-relative lateral distance and heading error are measured against the closest waypoint **on the planned route**, not against CARLA's own "nearest lane, regardless of direction" lookup — the latter is ambiguous inside any junction. Actions are exponentially smoothed (`α = 0.6`) before being applied, so per-step sampling noise from a stochastic policy does not translate into physically unrealistic steering oscillation.

### 2.2 Route planning and traffic-light compliance

A route between a randomly chosen start and destination is planned once per episode using CARLA's own road-topology graph (`GlobalRoutePlanner`, A* search). The vehicle's progress along that route is tracked every simulation step by a local search that walks forward from the last known position, tolerant of a bounded amount of backward drift (e.g. rolling back slightly on an incline) and immune to snapping onto a spatially closer but out-of-sequence point on a different street — a failure mode a naive nearest-point search exhibits on any reasonably dense street grid.

Traffic-light state is read from CARLA's own ground truth and enforced through the reward and termination logic: the agent is penalized for driving through a red light it had already begun stopping for, but is not penalized for the normal braking distance leading up to a light, nor for legitimately waiting at one — a distinction made by flagging a violation only at the moment the vehicle *exits* a light's trigger zone while still moving above a small speed threshold, rather than for the entire duration a light is red.

### 2.3 Reward function

At every simulation step, the agent receives:

```
r = w_center   · (1 − |lateral| / max_lateral)
  + w_speed    · exp(−(speed − target)² / 2σ²)
  + w_heading  · (1 − |heading_error| / π)
  + w_smooth   · max(0, 1 − Σ|Δaction| / 4)
  + w_progress · min(speed, target) / target
  + r_terminal + r_step
```

five shaped components — lane centering, a Gaussian speed target (peak at 30 km/h), heading alignment, action smoothness, and forward progress — plus a small constant per-step cost and a terminal penalty applied only on failure (never on a successful route completion or on a timeout). The forward-progress term exists specifically because a purely Gaussian speed reward has a small but non-zero value even at zero speed, which a deterministic policy (DDPG, TD3) can discover as a way to collect centering and smoothness reward indefinitely by simply not moving; the progress term is exactly zero at zero speed, closing that exploit.

### 2.4 Termination conditions

Evaluated in priority order — stall, collision, red-light violation, off-road, wrong heading, destination reached, timeout — so that a genuine failure always takes precedence over a coincident success, and vice versa a legitimate stop at a red light is never confused with stalling. The off-road threshold scales with the number of real same-direction lanes on the current road segment (derived from CARLA's own lane topology), so that drifting into an adjacent lane of a multi-lane one-way road — common near intersections, where dedicated turn lanes widen the road — is not treated identically to actually leaving the roadway.

### 2.5 Algorithms compared

Four continuous-control algorithms from Stable-Baselines3, trained against the same environment via a shared, algorithm-agnostic training pipeline:

| Algorithm | Policy | Exploration mechanism |
|---|---|---|
| PPO | On-policy, stochastic | Fixed entropy bonus |
| SAC | Off-policy, stochastic | Automatically-tuned entropy target |
| DDPG | Off-policy, deterministic | Ornstein–Uhlenbeck action noise |
| TD3 | Off-policy, deterministic | OU noise + clipped double-Q, target smoothing |

The two deterministic algorithms (DDPG, TD3) require both the forward-progress reward term above and a longer warm-up period of random exploration before learning begins, in order to escape the standing-still local optimum their exploration mechanism is otherwise prone to.

## 3. Results

All four algorithms were trained under an **equal 150,000-environment-step budget** and evaluated identically — 10 deterministic episodes per checkpoint. This is an intermediate result relative to the 500,000-step budget the project's configuration specifies as a final target (§5), but the comparison itself is fair: every algorithm sees the same number of environment interactions.

| Algorithm | Mean episode reward | Mean lateral distance | Success rate | Mean episode length |
|---|---|---|---|---|
| PPO | 617.87 ± 420.55 | 0.87 m | 10.0% | 224.6 steps |
| SAC | 1784.60 ± 973.86 | **0.56 m** | **40.0%** | 512.0 steps |
| DDPG | 1304.64 ± 1051.85 | 1.76 m | 30.0% | 424.4 steps |
| TD3 | **2613.62** ± 1066.66 | 0.56 m | 30.0% | **823.1 steps** |

No single algorithm dominates on every axis, and the comparison is more informative read along two separate questions than collapsed into one ranking:

**Which algorithm reaches a capable policy fastest?** Tracking the training step at which a 20-episode rolling-mean reward first sustains a "capable policy" threshold (2,500): only SAC crosses it within the 150k budget, at step 40,097 — roughly a quarter of the way through training. PPO, DDPG, and TD3 never sustain that level within the same budget.

**Which algorithm's final policy is most trustworthy?** By success rate, reward variance, and worst-case lateral excursion together: SAC is both the fastest to converge and the most reliable once trained. TD3 has the highest mean reward and by far the longest surviving episodes, but also the most run-to-run variance and a worse worst-case lateral deviation — a policy that occasionally drives very well but less consistently safely. DDPG is weakest specifically on control precision (worst mean and worst-case lateral distance). PPO has the lowest success rate of the four, but the tightest worst-case lateral control among all four when it does not succeed — its dominant failure mode is a rule-compliance one (driving through a red light) rather than a loss of lane control, a materially different kind of failure than the others'.

### 3.1 Comparison with published literature

An independent comparative study across the same four algorithms (plus TQC and CrossQ) on CARLA driving tasks, trained over 1,000,000 steps — roughly 6.7× this project's budget — reports the same relative ranking found here: SAC/TQC (off-policy, stochastic) achieving the best sample efficiency and task completion, DDPG the weakest, and PPO exhibiting visibly irregular training-reward trends attributed to its on-policy, replay-buffer-free updates — all independently reproduced in this project's own results and training curves. The absolute success rates reported here (10–40%) are lower than that study's (23–91% route completion) and lower still than affordance-based systems reporting up to 100% on simpler benchmarks; that gap is consistent with training at a fraction of the step budget and with using a compact 5-dimensional state rather than richer learned-affordance or sensor input, not evidence of a different underlying approach.

## 4. Figures

**Training curves — episode reward vs. environment steps, per algorithm:**

![Training curves](results/plots/training_curves.png)

**Head-to-head comparison — reward, success rate, lateral precision:**

![Comparison bars](results/plots/comparison_bars.png)

**Multi-metric comparison across six normalized performance dimensions:**

![Radar chart](results/plots/radar_chart.png)

**Termination-reason breakdown per algorithm's evaluation episodes:**

![Termination breakdown](results/plots/termination_breakdown.png)

**Sample efficiency — training step at which each algorithm first sustains a capable-policy reward level:**

![Sample efficiency](results/plots/sample_efficiency.png)

**Driven path vs. planned route** — an SAC policy's actual trajectory (colored by lateral deviation) against the planned route, for a genuine successful completion and a genuine failure:

![Trajectory — success](results/plots/trajectory_success_example.png)
![Trajectory — failure](results/plots/trajectory_failure_example.png)

**The simulation environment** — Town10HD_Opt, with an SAC policy actually driving:

![Environment overview](results/screenshots/01_map_overview.png)
![Route following](results/screenshots/02_route_lane_keeping.png)
![Intersection with an active red light](results/screenshots/03_intersection_redlight.png)

Additional figures (lateral-centering progress, evaluation score distributions, training stability, speed distribution, 3D trajectory views, further environment screenshots) are in `results/plots/` and `results/screenshots/`.

## 5. Limitations and future work

The 150,000-step comparison above is real and fairly conducted, but is an intermediate result relative to the 500,000-step budget the project's own configuration specifies as a final training length — whether to extend all four algorithms to that budget before finalizing thesis-quality numbers remains an open question. Traffic-light compliance is implemented against CARLA's own ground-truth signal state rather than a learned visual detector; the traffic-light and route-planning modules are both architecturally isolated behind single call sites specifically so a perception-based replacement can be substituted later without touching the reward, observation, or termination logic elsewhere. Stop-sign compliance, camera/lidar-based observation, and a discrete-action variant supporting DQN are designed for in the same modular shape but not yet implemented.

## 6. Technical details

| Component | Version |
|---|---|
| Simulator | CARLA 0.9.15 |
| RL library | Stable-Baselines3 2.0.0 |
| Environment interface | Gymnasium 0.28.1 |
| Neural network backend | PyTorch 1.13.1 |
| Language | Python 3.7.16 |

## References

- [CARLA Simulator](https://carla.org)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
- [Gymnasium](https://gymnasium.farama.org)
- Schulman et al., ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347), 2017
- Haarnoja et al., ["Soft Actor-Critic"](https://arxiv.org/abs/1801.01290), 2018
- Fujimoto et al., ["Addressing Function Approximation Error in Actor-Critic Methods"](https://arxiv.org/abs/1802.09477) (TD3), 2018
- Lillicrap et al., ["Continuous Control with Deep Reinforcement Learning"](https://arxiv.org/abs/1509.02971) (DDPG), 2015
- Alkhonain et al., ["A Comparative Study of Deep Reinforcement Learning Algorithms for Urban Autonomous Driving"](https://doi.org/10.3390/app15126838), Applied Sciences 15(12), 2025

# CARLA Reinforcement Learning — Lane Keeping Agent

A thesis-ready reinforcement learning project using CARLA 0.9.15 and PPO.
The agent learns to keep a vehicle in its lane using low-dimensional state observations.

---

## Project Structure

```
carla_rl_project/
├── carla_env/          # Gym environment (observation, action, reward)
├── agent/              # PPO training and evaluation scripts
├── scripts/            # Utility scripts (verify, spawn test, manual drive)
├── configs/            # YAML configuration files
├── results/            # Logs, checkpoints, plots (generated at runtime)
├── docs/               # Architecture notes for thesis writing
├── requirements.txt
└── README.md
```

---

## System Requirements

- Ubuntu 22.04
- CARLA 0.9.15
- Python 3.8 (recommended for CARLA API compatibility)
- GPU with at least 6GB VRAM (for running CARLA server)

---

## Setup Instructions

### Step 1 — Clone or create the project folder

```bash
cd ~
git clone <your-repo-url> carla_rl_project
cd carla_rl_project
```

### Step 2 — Create a Python virtual environment

```bash
python3.8 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Add the CARLA Python API to your environment

CARLA ships its own Python egg file. You need to point Python to it.
Replace `/path/to/carla` with your actual CARLA installation path.

```bash
# Option A: Add to your virtual environment permanently
echo "export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg" >> venv/bin/activate

# Option B: Add to your .bashrc (affects all terminals)
echo "export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg" >> ~/.bashrc
source ~/.bashrc
```

To verify the CARLA egg is reachable:
```bash
python -c "import carla; print('CARLA API version:', carla.__version__)"
```

### Step 5 — Launch the CARLA server

Open a separate terminal and run:
```bash
cd /path/to/carla
./CarlaUE4.sh -quality-level=Low -fps=20
```

Flags explained:
- `-quality-level=Low` — faster rendering, good for training
- `-fps=20` — cap server FPS (we use synchronous mode so this is a safety cap)

### Step 6 — Verify the connection

With CARLA running, in your project terminal:
```bash
python scripts/verify_carla.py
```

Expected output:
```
[INFO] Connecting to CARLA at localhost:2000 ...
[INFO] Connected. Server version: 0.9.15
[INFO] Loading map: Town03 ...
[INFO] Map loaded: Town03
[INFO] Available maps: [list of maps]
[INFO] Number of spawn points: <N>
[INFO] Weather: ClearNoon
[INFO] Verification complete. CARLA is ready.
```

---

## Running the Project (later phases)

```bash
# Manual driving test
python scripts/manual_drive.py

# Train the PPO agent
python agent/train.py --config configs/config.yaml

# Evaluate a trained agent
python agent/evaluate.py --checkpoint results/checkpoints/best_model
```

---

## Thesis Notes

- The environment follows the OpenAI Gymnasium interface (`reset`, `step`, `render`).
- Synchronous mode is used throughout for reproducibility.
- All hyperparameters are stored in `configs/config.yaml`.
- Training curves are logged to `results/logs/` and viewable with TensorBoard.

---

## References

- CARLA Simulator: https://carla.org
- Stable-Baselines3: https://stable-baselines3.readthedocs.io
- Gymnasium: https://gymnasium.farama.org
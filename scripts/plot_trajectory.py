"""
plot_trajectory.py
-------------------
Drive one live episode with a trained checkpoint and plot the vehicle's
actual (x, y) path against the planned route -- shows *where* the agent
deviates from the route, which the aggregate metrics (mean_lateral_dist)
can't. No training or checkpoint changes involved; the model is only
used for inference (model.predict()).

Drives up to MAX_EPISODES episodes looking for one genuine success
(destination_reached) and one genuine failure (any other terminal
reason) to plot side by side -- more honest than cherry-picking a
single "looks nice" run, and SAC's real ~40% success rate (see
docs/THESIS_CONTEXT.md sec. 5) means a clean success often takes a few
tries. Produces up to 4 figures in results/plots/:
    trajectory_success_example.png / _3d.png
    trajectory_failure_example.png / _3d.png

Run with CARLA already started:
    python scripts/plot_trajectory.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d projection
import numpy as np

OUTPUT_DIR = "results/plots"
SAC_CHECKPOINT = "results/checkpoints/sac/sac_lane_keeping_20260908_190524/final_model.zip"
MAX_STEPS = 900
MAX_EPISODES = 6   # how many episodes to try before giving up on finding one of each


def log(msg):
    print("[INFO] {}".format(msg))


def drive_and_record(env, model):
    """
    Drive one episode with the given model and record per-step vehicle
    position + lateral deviation. Returns a dict of parallel lists.
    """
    obs, info = env.reset()
    route_xy = [(wp.waypoint.transform.location.x, wp.waypoint.transform.location.y) for wp in env._route]

    xs, ys, laterals = [], [], []
    reason = "timeout"
    for step in range(MAX_STEPS):
        loc = env._vehicle.get_transform().location
        xs.append(loc.x)
        ys.append(loc.y)
        laterals.append(abs(info.get("lateral_distance", 0.0)))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            reason = info.get("termination_reason")
            log("Episode ended at step {} (reason: {})".format(step, reason))
            break

    return {"route_x": [p[0] for p in route_xy], "route_y": [p[1] for p in route_xy],
            "x": xs, "y": ys, "lateral": laterals, "termination_reason": reason}


def plot_top_down(data, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(data["route_x"], data["route_y"], "--", color="#999999", linewidth=1.5,
            label="Planned route", zorder=1)

    sc = ax.scatter(data["x"], data["y"], c=data["lateral"], cmap="RdYlGn_r",
                     vmin=0, vmax=1.5, s=14, zorder=2, label="Driven path")
    ax.plot(data["x"][0], data["y"][0], marker="o", color="black", markersize=10,
            zorder=3, label="Start")
    ax.plot(data["x"][-1], data["y"][-1], marker="s", color="black", markersize=10,
            zorder=3, label="End")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Lateral distance from route (m)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Driven Path vs. Planned Route (SAC, deterministic policy)", fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log("saved: {}".format(path))


def plot_3d(data, path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    t = list(range(len(data["x"])))
    sc = ax.scatter(data["x"], data["y"], t, c=data["lateral"], cmap="RdYlGn_r",
                     vmin=0, vmax=1.5, s=14)
    ax.plot(data["route_x"], data["route_y"], zs=0, zdir="z", linestyle="--",
            color="#999999", linewidth=1.2, label="Planned route (t=0 plane)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Simulation step")
    ax.set_title("Driven Path Over Time (SAC, deterministic policy)", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Lateral distance from route (m)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log("saved: {}".format(path))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from carla_env.env import CarlaLaneKeepingEnv
    from agent.evaluate import load_model

    log("Loading SAC checkpoint: {}".format(SAC_CHECKPOINT))
    model = load_model("sac", SAC_CHECKPOINT)

    success, failure = None, None
    for attempt in range(MAX_EPISODES):
        env = CarlaLaneKeepingEnv(render_debug=False, map_name="Town10HD_Opt")
        try:
            data = drive_and_record(env, model)
        finally:
            env.close()

        reason = data["termination_reason"]
        log("Episode {}/{}: {} steps, reason={}".format(attempt + 1, MAX_EPISODES, len(data["x"]), reason))

        if reason == "destination_reached" and len(data["x"]) > 50 and success is None:
            success = data
        elif reason not in ("destination_reached", "timeout") and failure is None:
            failure = data

        if success is not None and failure is not None:
            break

    if success is not None:
        plot_top_down(success, os.path.join(OUTPUT_DIR, "trajectory_success_example.png"))
        plot_3d(success, os.path.join(OUTPUT_DIR, "trajectory_success_example_3d.png"))
    else:
        log("WARNING: no destination_reached episode found in {} attempts -- skipping success plot".format(MAX_EPISODES))

    if failure is not None:
        plot_top_down(failure, os.path.join(OUTPUT_DIR, "trajectory_failure_example.png"))
        plot_3d(failure, os.path.join(OUTPUT_DIR, "trajectory_failure_example_3d.png"))
    else:
        log("WARNING: no failure episode found in {} attempts -- skipping failure plot".format(MAX_EPISODES))

    log("Done.")


if __name__ == "__main__":
    main()

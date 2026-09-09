"""
capture_screenshots.py
-----------------------
Grab a small, curated set of representative screenshots from the live
CARLA environment for the thesis's "environment" section. Purely visual
context for the reader (what the world looks like) -- not a data
artifact, no metrics involved.

A trained SAC checkpoint drives the car so the driving shots show
competent, on-policy behaviour rather than random/jerky motion.

Run with CARLA already started:
    python scripts/capture_screenshots.py

Images are written to results/screenshots/:
    01_map_overview.png       top-down view of the whole map
    02_route_lane_keeping.png chase view, straight road, route overlay
    03_intersection_redlight.png chase view, stopped at a red light
    04_driver_pov.png         windshield-height forward view
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/screenshots"
SAC_CHECKPOINT = "results/checkpoints/sac/sac_lane_keeping_20260908_190524/final_model.zip"
MAX_DRIVE_STEPS = 400   # cap on how long we'll drive looking for a red light


def log(msg):
    print("[INFO] {}".format(msg))


def _spawn_rgb_camera(world, transform, attach_to=None, width=1280, height=720, fov=100):
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    return world.spawn_actor(bp, transform, attach_to=attach_to)


def _save_camera_frame(world, camera, path):
    """
    Tick and save a frame from this camera, discarding the very first one.
    A camera's first tick right after spawning can render with partial
    HUD/black-bar artifacts before the pipeline settles -- confirmed by
    inspecting an early capture from this script.
    """
    captured = {"count": 0}

    def _on_image(image):
        captured["count"] += 1
        captured["image"] = image

    camera.listen(_on_image)
    for _ in range(6):
        world.tick()
        if captured["count"] >= 2:
            break
    camera.stop()
    if "image" in captured:
        captured["image"].save_to_disk(path)
        log("saved: {}".format(path))
    else:
        log("WARNING: no frame captured for {}".format(path))


def capture_topdown(world, path):
    import carla

    spawn_points = world.get_map().get_spawn_points()
    xs = [p.location.x for p in spawn_points]
    ys = [p.location.y for p in spawn_points]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    transform = carla.Transform(
        carla.Location(x=cx, y=cy, z=300),
        carla.Rotation(pitch=-90, yaw=0, roll=0),
    )
    camera = _spawn_rgb_camera(world, transform, attach_to=None, width=1600, height=1600, fov=100)
    try:
        _save_camera_frame(world, camera, path)
    finally:
        camera.destroy()


def capture_chase(world, vehicle, path, distance_m=7.0, height_m=3.2, pitch_deg=-12.0, fov=100):
    from carla_env.env import _compute_spectator_transform

    transform = _compute_spectator_transform(vehicle.get_transform(), distance_m, height_m, pitch_deg)
    camera = _spawn_rgb_camera(world, transform, attach_to=None, fov=fov)
    try:
        _save_camera_frame(world, camera, path)
    finally:
        camera.destroy()


def capture_driver_pov(world, vehicle, path):
    import carla

    relative = carla.Transform(carla.Location(x=0.8, y=-0.3, z=1.3), carla.Rotation(pitch=-3))
    camera = _spawn_rgb_camera(world, relative, attach_to=vehicle, fov=100)
    try:
        _save_camera_frame(world, camera, path)
    finally:
        camera.destroy()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from carla_env.env import CarlaLaneKeepingEnv
    from agent.evaluate import load_model

    import carla
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    log("1/4 - top-down map overview")
    capture_topdown(world, os.path.join(OUTPUT_DIR, "01_map_overview.png"))

    log("Loading SAC checkpoint to drive the demo episode: {}".format(SAC_CHECKPOINT))
    model = load_model("sac", SAC_CHECKPOINT)

    # CarlaLaneKeepingEnv's own default map (Town03) segfaults this
    # machine's CARLA install (a known, out-of-scope engine/GPU-driver
    # issue -- see docs/THESIS_CONTEXT.md's limitations section).
    # Town10HD_Opt, the map configs/config.yaml actually trains on, is
    # already loaded from the top-down shot above -- reuse it explicitly.
    env = CarlaLaneKeepingEnv(render_debug=True, map_name="Town10HD_Opt")
    obs, info = env.reset()

    log("2/4 - driving a few steps for the route/lane-keeping shot")
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    capture_chase(world, env._vehicle, os.path.join(OUTPUT_DIR, "02_route_lane_keeping.png"))

    log("3/4 - driving toward a red light (up to {} steps)".format(MAX_DRIVE_STEPS))
    found_red_light = False
    for step in range(MAX_DRIVE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("traffic_light_must_stop"):
            found_red_light = True
            # Narrower FOV than the other chase shots -- zooms in enough
            # that the traffic light heads down the road are actually
            # legible instead of tiny background detail.
            capture_chase(
                world, env._vehicle,
                os.path.join(OUTPUT_DIR, "03_intersection_redlight.png"),
                distance_m=9.0, height_m=3.5, pitch_deg=-5.0, fov=45,
            )
            break
        if terminated or truncated:
            obs, info = env.reset()
    if not found_red_light:
        log("WARNING: never hit a red light within {} steps -- skipping shot 3".format(MAX_DRIVE_STEPS))

    log("4/4 - driver POV shot")
    capture_driver_pov(world, env._vehicle, os.path.join(OUTPUT_DIR, "04_driver_pov.png"))

    env.close()
    log("Done. Screenshots in {}/".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()

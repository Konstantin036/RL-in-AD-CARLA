"""
manual_drive.py
---------------
Phase 7 — Manual Driving Script

Purpose:
    Drive the ego vehicle manually using the keyboard.
    This lets you verify the environment visually before training:
        - Does the vehicle handle correctly?
        - Do observation values change as expected?
        - Does the reward respond to good/bad driving?
        - Does the episode reset cleanly after going off road?

Controls:
    W / Up    → throttle
    S / Down  → brake / reverse
    A / Left  → steer left
    D / Right → steer right
    R         → reset episode manually
    Q / ESC   → quit

Requirements:
    pip install pygame

How to run:
    1. Launch CARLA:  ./CarlaUE4.sh -quality-level=Low -fps=20
    2. Run this:      python scripts/manual_drive.py
    3. Click the pygame window to give it keyboard focus
    4. Drive with WASD or arrow keys

What to watch in the terminal:
    - lateral_distance should change as you drift left/right
    - heading_error should change as you go through curves
    - reward should be high when centered and aligned
    - episode resets automatically on collision or off-road
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Pygame display constants ───────────────────────────────────────────────────

WINDOW_WIDTH  = 500
WINDOW_HEIGHT = 300
FPS_TARGET    = 20   # match CARLA's sync rate


# ── HUD renderer ──────────────────────────────────────────────────────────────

class HUD:
    """
    Draws a simple heads-up display in the pygame window showing
    the current observation values, reward, and controls.
    """

    def __init__(self, screen, font, font_small):
        self.screen     = screen
        self.font       = font
        self.font_small = font_small

        # Colors
        self.BLACK  = (0,   0,   0)
        self.WHITE  = (255, 255, 255)
        self.GREEN  = (50,  200, 50)
        self.YELLOW = (220, 200, 50)
        self.RED    = (220, 60,  60)
        self.GRAY   = (120, 120, 120)
        self.BLUE   = (80,  140, 220)

    def _color_for_value(self, value, threshold_warn, threshold_bad):
        """Return green/yellow/red based on how far a value is from zero."""
        abs_val = abs(value)
        if abs_val < threshold_warn:
            return self.GREEN
        elif abs_val < threshold_bad:
            return self.YELLOW
        else:
            return self.RED

    def render(self, obs_data, reward_info, step, episode,
               throttle, brake, steer, terminated, truncated, term_reason):
        """Draw the full HUD onto the pygame screen."""
        self.screen.fill((20, 20, 20))   # dark background

        y = 20
        line_h = 28

        def text(msg, x, yy, color=None, big=False):
            f = self.font if big else self.font_small
            surf = f.render(msg, True, color or self.WHITE)
            self.screen.blit(surf, (x, yy))

        # Title
        text("CARLA Lane Keeping — Manual Drive", 20, y, self.BLUE, big=True)
        y += 38

        # Episode / step
        text(f"Episode: {episode}    Step: {step}", 20, y, self.GRAY)
        y += line_h

        # ── Observation values ─────────────────────────────────────────────────
        text("── Observation ──", 20, y, self.GRAY)
        y += line_h

        lat   = obs_data.lateral_distance_m
        hdg   = math.degrees(obs_data.heading_error_rad)
        spd   = obs_data.speed_kmh
        steer_val = obs_data.steering

        lat_color = self._color_for_value(lat,   0.5, 2.0)
        hdg_color = self._color_for_value(hdg,  10.0, 45.0)
        spd_color = self.GREEN if 10 < spd < 50 else self.YELLOW

        text(f"Lateral dist:  {lat:+.3f} m", 30, y, lat_color)
        y += line_h
        text(f"Heading error: {hdg:+.1f} °", 30, y, hdg_color)
        y += line_h
        text(f"Speed:         {spd:.1f} km/h", 30, y, spd_color)
        y += line_h
        text(f"Steering:      {steer_val:+.2f}", 30, y)
        y += line_h + 4

        # ── Reward breakdown ───────────────────────────────────────────────────
        text("── Reward ──", 20, y, self.GRAY)
        y += line_h

        r_color = self.GREEN if reward_info.total > 1.0 else (
                  self.YELLOW if reward_info.total > 0 else self.RED)

        text(f"Total:    {reward_info.total:+.3f}", 30, y, r_color, big=True)
        y += 32
        text(f"Center={reward_info.r_centering:.3f}  "
             f"Speed={reward_info.r_speed:.3f}  "
             f"Heading={reward_info.r_heading:.3f}", 30, y, self.GRAY)
        y += line_h + 4

        # ── Controls ───────────────────────────────────────────────────────────
        text("── Controls ──", 20, y, self.GRAY)
        y += line_h

        thr_color = self.GREEN  if throttle > 0.1 else self.GRAY
        brk_color = self.RED    if brake    > 0.1 else self.GRAY
        str_color = self.YELLOW if abs(steer) > 0.05 else self.GRAY

        text(f"Throttle: {throttle:.2f}", 30, y, thr_color)
        y += line_h
        text(f"Brake:    {brake:.2f}",    30, y, brk_color)
        y += line_h
        text(f"Steer:    {steer:+.2f}",   30, y, str_color)
        y += line_h + 4

        # ── Episode end message ────────────────────────────────────────────────
        if terminated or truncated:
            end_msg = f"Episode ended: {term_reason}  —  Press R to reset"
            text(end_msg, 20, y, self.RED, big=True)
            y += 32

        # ── Key hints at bottom ────────────────────────────────────────────────
        hints = "WASD / arrows = drive   R = reset   Q/ESC = quit"
        text(hints, 20, WINDOW_HEIGHT - 28, self.GRAY)

        import pygame
        pygame.display.flip()


# ── Main manual drive loop ─────────────────────────────────────────────────────

def run_manual_drive():
    """
    Main loop: pygame window + CARLA environment + keyboard control.
    """

    # ── Import pygame ──────────────────────────────────────────────────────────
    try:
        import pygame
    except ImportError:
        print("[ERROR] pygame is not installed.")
        print("        Run:  pip install pygame")
        sys.exit(1)

    from carla_env.env    import CarlaLaneKeepingEnv
    from carla_env.reward import RewardConfig, compute_reward
    from carla_env.observation import compute_observation

    # ── Init pygame ────────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("CARLA Lane Keeping — Manual Drive")
    clock  = pygame.time.Clock()

    font       = pygame.font.SysFont("monospace", 16, bold=True)
    font_small = pygame.font.SysFont("monospace", 14)

    hud = HUD(screen, font, font_small)

    # ── Create environment ─────────────────────────────────────────────────────
    print("[INFO] Creating environment ...")
    env = CarlaLaneKeepingEnv(
        host      = "localhost",
        port      = 2000,
        map_name  = "Town03",
        max_steps = 2000,     # long episodes for manual testing
        verbose   = False,
    )
    reward_cfg = RewardConfig()

    # ── Episode state ──────────────────────────────────────────────────────────
    obs, info     = env.reset()
    terminated    = False
    truncated     = False
    term_reason   = ""
    step          = 0
    episode       = 1

    # Manual control values (updated from keyboard each frame)
    throttle = 0.0
    brake    = 0.0
    steer    = 0.0

    # Dummy initial obs_data and reward_info for first frame
    from carla_env.observation import ObservationData
    from carla_env.reward      import RewardInfo
    import math as _math
    obs_data    = ObservationData(0.0, 0.0, 0.0, 0.0)
    reward_info = RewardInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    print("[INFO] Manual drive started. Click the pygame window and use WASD to drive.")
    print("[INFO] Press R to reset, Q or ESC to quit.")

    running = True
    while running:

        # ── Process pygame events ──────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_r:
                    # Manual reset
                    obs, info  = env.reset()
                    terminated = False
                    truncated  = False
                    term_reason = ""
                    step       = 0
                    episode   += 1
                    obs_data    = ObservationData(0.0, 0.0, 0.0, 0.0)
                    reward_info = RewardInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
                    print(f"[INFO] Manual reset → Episode {episode}")

        # ── Read keyboard state ────────────────────────────────────────────────
        keys = pygame.key.get_pressed()

        throttle = 0.0
        brake    = 0.0
        steer    = 0.0

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            throttle = 0.7
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            brake    = 0.5
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer    = -0.5
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer    =  0.5

        # ── Step environment ───────────────────────────────────────────────────
        if not (terminated or truncated):
            # Convert keyboard inputs to action array
            # acceleration = throttle - brake  (maps to our [-1,1] scale)
            acceleration = throttle - brake
            action = np.array([acceleration, steer], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            term_reason = info.get("termination_reason", "")

            # Get rich obs_data and reward_info for HUD
            # We re-compute from the env's internal vehicle state
            from carla_env.observation import ObservationData
            obs_data = ObservationData(
                lateral_distance_m = info["lateral_distance"],
                heading_error_rad  = _math.radians(info["heading_error_deg"]),
                speed_kmh          = info["speed_kmh"],
                steering           = info["steering"],
            )

            from carla_env.reward import RewardInfo
            reward_info = RewardInfo(
                total       = info["reward_total"],
                r_centering = info["reward_centering"],
                r_speed     = info["reward_speed"],
                r_heading   = info["reward_heading"],
                r_terminal  = info["reward_terminal"],
                r_step      = -0.05,
                is_terminal = terminated,
            )

            # Auto-reset on episode end
            if terminated or truncated:
                print(f"[INFO] Episode {episode} ended: {term_reason} "
                      f"(step {step}). Press R to start next episode.")

        # ── Render HUD ─────────────────────────────────────────────────────────
        hud.render(
            obs_data    = obs_data,
            reward_info = reward_info,
            step        = step,
            episode     = episode,
            throttle    = throttle,
            brake       = brake,
            steer       = steer,
            terminated  = terminated,
            truncated   = truncated,
            term_reason = term_reason,
        )

        clock.tick(FPS_TARGET)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    print("[INFO] Quitting ...")
    env.close()
    pygame.quit()
    print("[INFO] Done.")


if __name__ == "__main__":
    run_manual_drive()

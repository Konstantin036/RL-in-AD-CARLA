"""
sensors.py
----------
Phase 6 — Sensor Management

Purpose:
    Manage CARLA sensors attached to the ego vehicle.
    For now we only need the collision sensor — it fires a callback
    whenever the vehicle makes physical contact with anything.

    Camera and lidar sensors can be added here later (Phase extension).

How CARLA sensors work:
    1. Create a sensor blueprint
    2. Attach it to the vehicle with a relative transform
    3. Register a callback function — CARLA calls it asynchronously
       whenever the sensor fires
    4. In synchronous mode, the callback fires during world.tick()
    5. Destroy the sensor when the episode ends

Why a separate file?
    Sensor lifecycle (create, attach, listen, destroy) is verbose enough
    that mixing it into env.py makes that file hard to read. Isolating it
    here keeps env.py focused on the RL logic.
"""


class CollisionSensor:
    """
    Attaches a collision sensor to a vehicle and tracks collision events.

    Usage in env.py:
        # At episode start (in reset):
        self.collision_sensor = CollisionSensor(world, vehicle)

        # At each step:
        if self.collision_sensor.has_collided:
            terminated = True

        # At episode end (in reset or close):
        self.collision_sensor.destroy()

    The sensor works by registering a callback with CARLA. When CARLA
    detects a collision during world.tick(), it calls our _on_collision()
    method, which sets the has_collided flag to True.

    The flag is never automatically reset — you must call destroy() and
    create a new CollisionSensor each episode, or call reset_flag() if
    you reuse the sensor across episodes.
    """

    def __init__(self, world, vehicle):
        """
        Create and attach a collision sensor to the vehicle.

        Args:
            world:   carla.World
            vehicle: carla.Vehicle — the ego vehicle to attach to
        """
        self.has_collided = False
        self._sensor = None

        # Find the collision sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Attach at the vehicle's center, no offset needed for collision detection
        # carla.Transform() with no arguments = identity (zero offset, zero rotation)
        import carla
        sensor_transform = carla.Transform()

        # spawn_actor with attach_to= places the sensor relative to the vehicle.
        # The sensor will follow the vehicle automatically — we never need to
        # move it manually.
        self._sensor = world.spawn_actor(bp, sensor_transform, attach_to=vehicle)

        # Register the callback. CARLA calls this during world.tick() if a
        # collision is detected. The lambda just routes to our method.
        self._sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event) -> None:
        """
        Callback fired by CARLA when the vehicle collides with anything.

        Args:
            event: carla.CollisionEvent
                   event.actor          = the vehicle that collided
                   event.other_actor    = what it hit
                   event.normal_impulse = force vector of the collision

        We only care that a collision happened, not the details.
        The flag is checked in env.py's step() method.
        """
        self.has_collided = True

    def reset_flag(self) -> None:
        """Reset the collision flag without destroying the sensor."""
        self.has_collided = False

    def destroy(self) -> None:
        """
        Stop the sensor and remove it from the CARLA world.
        Always call this before destroying the vehicle.

        If the sensor is destroyed after the vehicle, CARLA may crash
        or leave ghost sensors in the world. Always destroy sensors first.
        """
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()      # stop the callback listener first
            self._sensor.destroy()   # then remove from world
            self._sensor = None

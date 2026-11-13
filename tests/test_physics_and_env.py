import math
import pybullet as p
import pytest

from simulation.env import RoboTeachEnv
from simulation.camera import SimCamera
from perception.vision_locator import VisionLocator
from skills.primitives import RobotSkills


@pytest.fixture
def sim_ctx():
    env = RoboTeachEnv(headless=True, seed=42, include_container=True, include_obstacle=False)
    camera = SimCamera(headless=True)
    locator = VisionLocator(camera, env.object_dimensions)
    skills = RobotSkills(env.robot_id, env.objects, locator, control_mode="pd", realtime=False)
    skills.home()
    yield env, camera, locator, skills
    env.close()


def test_deterministic_seeded_randomization():
    env = RoboTeachEnv(headless=True, seed=123, include_container=True, include_obstacle=True)
    try:
        assert "brown_box" in env.objects
        assert "obstacle_wall" in env.objects
        env.randomize(seed=99)
        pos_a = {k: p.getBasePositionAndOrientation(v)[0] for k, v in env.objects.items()}
        env.reset()
        env.randomize(seed=99)
        pos_b = {k: p.getBasePositionAndOrientation(v)[0] for k, v in env.objects.items()}
        for name in ("red_cube", "blue_cube", "green_cube", "yellow_cube"):
            dist = math.dist(pos_a[name], pos_b[name])
            assert dist < 1e-3, f"Non-deterministic position for {name}: {dist}"
    finally:
        env.close()


def test_hollow_container_accepts_cube_inside():
    env = RoboTeachEnv(headless=True, seed=7, include_container=True)
    try:
        box_pos, _ = p.getBasePositionAndOrientation(env.objects["brown_box"])
        red_id = env.objects["red_cube"]
        # Drop red_cube directly above container center
        p.resetBasePositionAndOrientation(red_id, [box_pos[0], box_pos[1], box_pos[2] + 0.08], [0, 0, 0, 1])
        for _ in range(240):
            p.stepSimulation()
        final_pos, _ = p.getBasePositionAndOrientation(red_id)
        assert abs(final_pos[0] - box_pos[0]) < 0.05
        assert abs(final_pos[1] - box_pos[1]) < 0.05
        # Must settle inside cavity (below rim z ~ 0.49, above table 0.42)
        assert 0.425 <= final_pos[2] <= 0.475
    finally:
        env.close()


def test_pd_contact_grasp_and_two_cube_stack(sim_ctx):
    env, _, locator, skills = sim_ctx
    # Place red and blue at known reachable workspace locations
    p.resetBasePositionAndOrientation(env.objects["blue_cube"], [0.16, -0.08, 0.445], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(env.objects["red_cube"], [0.16, 0.08, 0.445], [0, 0, 0, 1])
    for _ in range(120):
        p.stepSimulation()
    locator.clear_cache()

    assert skills.grasp("red_cube") is True
    # Must NOT use fake weld constraint in PD mode
    assert skills.constraint_id is None
    assert skills.is_holding_object() is True

    blue_pos = skills.get_object_position("blue_cube")
    assert blue_pos is not None
    skills.move_to([blue_pos[0], blue_pos[1], blue_pos[2] + 0.12])
    skills.move_to([blue_pos[0], blue_pos[1], blue_pos[2] + 0.053])
    assert skills.release() is True
    skills.home()

    for _ in range(180):
        p.stepSimulation()

    red_gt, _ = p.getBasePositionAndOrientation(env.objects["red_cube"])
    blue_gt, _ = p.getBasePositionAndOrientation(env.objects["blue_cube"])
    xy_err = math.hypot(red_gt[0] - blue_gt[0], red_gt[1] - blue_gt[1])
    assert xy_err < 0.025
    assert red_gt[2] - blue_gt[2] > 0.042


import math
import pybullet as p

from simulation.env import RoboTeachEnv
from simulation.camera import SimCamera
from perception.vision_locator import VisionLocator
from skills.primitives import RobotSkills


def test_multiview_rgbd_localization_accuracy_and_container():
    env = RoboTeachEnv(headless=True, seed=11, include_container=True, include_obstacle=True)
    try:
        camera = SimCamera(headless=True)
        locator = VisionLocator(camera, env.object_dimensions)
        skills = RobotSkills(env.robot_id, env.objects, locator, control_mode="pd", realtime=False)
        skills.home()

        detected = locator.locate_all()
        for name in ("red_cube", "blue_cube", "green_cube", "yellow_cube", "brown_box", "obstacle_wall"):
            assert name in detected, f"Missing {name} from camera detection"
            gt_pos, _ = p.getBasePositionAndOrientation(env.objects[name])
            est_pos = detected[name]
            xy_err = math.hypot(est_pos[0] - gt_pos[0], est_pos[1] - gt_pos[1])
            assert xy_err < 0.025, f"XY localization error too large for {name}: {xy_err:.4f}m"
    finally:
        env.close()


def test_occluded_bottom_cube_located_via_lateral_views():
    env = RoboTeachEnv(headless=True, seed=21, include_container=False)
    try:
        camera = SimCamera(headless=True)
        locator = VisionLocator(camera, env.object_dimensions)
        skills = RobotSkills(env.robot_id, env.objects, locator, control_mode="pd", realtime=False)
        skills.home()

        # Stack red directly on blue so top view cannot see blue_cube at all
        p.resetBasePositionAndOrientation(env.objects["blue_cube"], [0.16, 0.0, 0.445], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(env.objects["red_cube"], [0.16, 0.0, 0.496], [0, 0, 0, 1])
        for _ in range(120):
            p.stepSimulation()

        locator.clear_cache()
        blue_est = locator.locate("blue_cube")
        assert blue_est is not None, "Occluded bottom cube should be detected by lateral isometric views"
        blue_gt, _ = p.getBasePositionAndOrientation(env.objects["blue_cube"])
        assert math.dist(blue_est, blue_gt) < 0.03
    finally:
        env.close()


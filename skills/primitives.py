# primitives.py — Franka Panda Manipulation Skills (True PD Motor Control & Contact Grasping)
import math
import time
import pybullet as p


class RobotSkills:
    """
    Low-level manipulation skills for 7-DOF Franka Emika Panda.
    Supports:
      - control_mode="pd" (default): Minimum-jerk quintic joint-space trajectory tracking via
        POSITION_CONTROL motors and genuine frictional parallel-jaw contact grasping (zero weld
        constraints, zero cube-pose teleportation).
      - control_mode="kinematic": Legacy state-teleport baseline retained strictly for ablation.
    """

    def __init__(
        self,
        robot_id: int,
        objects: dict[str, int],
        vision_locator,
        end_effector_index: int = 11,
        control_mode: str = "pd",
        realtime: bool = False,
        physics_client_id: int = 0,
    ):
        self.robot_id = robot_id
        self.objects = objects
        self.vision_locator = vision_locator
        self.ee_link = end_effector_index
        self.num_joints = 7
        self.control_mode = control_mode
        self.realtime = realtime
        self.client = physics_client_id

        self.grasped_object: int | None = None
        self.grasped_name: str | None = None
        self.constraint_id: int | None = None

        self.lower_limits = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        self.upper_limits = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        self.joint_ranges = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9096, 5.9342]
        self.rest_poses = [-1.2, 0.0, 0.0, -1.5708, 0.0, 1.8675, -0.7854]
        self.grip_orn = p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2.0])

        # Calibratable kinematic execution parameters (updated by CalibrationMemory)
        self.policy_params = {
            "transit_z": 0.64,
            "release_margin_z": 0.052,
            "grasp_force": 75.0,
            "auto_safe_lift": True,
        }

    def configure_params(self, params: dict | None):
        if params:
            self.policy_params.update(params)

    def _step_sim(self, steps: int = 1):
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client)
            if self.realtime:
                time.sleep(1.0 / 240.0)

    def get_safe_z(self) -> float:
        """Computes safe transit height purely from perceived 3D scene bounds (never ground-truth)."""
        base_z = float(self.policy_params.get("transit_z", 0.64))
        if not self.policy_params.get("auto_safe_lift", True):
            return base_z
        details = getattr(self.vision_locator, "_last_metadata", None)
        if not details or len(details) < 4:
            details = self.vision_locator.locate_all_detailed()
        max_top = 0.47
        for info in details.values():
            max_top = max(max_top, float(info.get("z_top", 0.47)))
        return max(base_z, min(0.82, max_top + 0.11))

    def _solve_ik(self, position: list[float]) -> list[float]:
        angles = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            position,
            self.grip_orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_poses,
            maxNumIterations=600,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        return list(angles[: self.num_joints])

    def _get_joint_positions(self) -> list[float]:
        return [
            p.getJointState(self.robot_id, i, physicsClientId=self.client)[0]
            for i in range(self.num_joints)
        ]

    def _hold_gripper_command(self):
        if self.grasped_object is not None and self.control_mode == "pd":
            force = float(self.policy_params.get("grasp_force", 55.0))
            p.setJointMotorControl2(
                self.robot_id, 9, p.POSITION_CONTROL, 0.0, force=force, physicsClientId=self.client
            )
            p.setJointMotorControl2(
                self.robot_id, 10, p.POSITION_CONTROL, 0.0, force=force, physicsClientId=self.client
            )

    def _move_joints(self, target_angles: list[float], duration_steps: int = 160):
        current = self._get_joint_positions()
        if self.control_mode == "kinematic":
            waypoints = 75
            for wp in range(1, waypoints + 1):
                t = wp / waypoints
                for i in range(self.num_joints):
                    ang = current[i] + (target_angles[i] - current[i]) * t
                    p.resetJointState(self.robot_id, i, ang, physicsClientId=self.client)
                self._step_sim(2)
            return

        # True PD motor tracking along minimum-jerk quintic profile: s(t) = 10t^3 - 15t^4 + 6t^5
        joint_indices = list(range(self.num_joints))
        forces = [240.0, 240.0, 240.0, 240.0, 140.0, 140.0, 140.0]
        kp = [0.28] * self.num_joints
        kd = [1.05] * self.num_joints

        for step in range(1, duration_steps + 1):
            tau = step / duration_steps
            s = tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau))
            desired = [current[i] + (target_angles[i] - current[i]) * s for i in range(self.num_joints)]
            p.setJointMotorControlArray(
                self.robot_id,
                joint_indices,
                p.POSITION_CONTROL,
                targetPositions=desired,
                forces=forces,
                positionGains=kp,
                velocityGains=kd,
                physicsClientId=self.client,
            )
            self._hold_gripper_command()
            self._step_sim(1)

        # Settle at target setpoint
        for _ in range(35):
            p.setJointMotorControlArray(
                self.robot_id,
                joint_indices,
                p.POSITION_CONTROL,
                targetPositions=target_angles,
                forces=forces,
                positionGains=kp,
                velocityGains=kd,
                physicsClientId=self.client,
            )
            self._hold_gripper_command()
            self._step_sim(1)

    def _ik_move(self, target_pos: list[float], duration_steps: int = 150) -> list[float]:
        target_angles = self._solve_ik(target_pos)
        self._move_joints(target_angles, duration_steps=duration_steps)
        return self.get_ee_pos()

    def get_ee_pos(self) -> list[float]:
        state = p.getLinkState(self.robot_id, self.ee_link, physicsClientId=self.client)
        return list(state[0])

    def open_gripper(self, slow: bool = False):
        steps = 90 if slow else 50
        cur_l = p.getJointState(self.robot_id, 9, physicsClientId=self.client)[0]
        for s in range(1, steps + 1):
            target = cur_l + (0.04 - cur_l) * (s / steps)
            p.setJointMotorControl2(
                self.robot_id, 9, p.POSITION_CONTROL, target, force=80.0, physicsClientId=self.client
            )
            p.setJointMotorControl2(
                self.robot_id, 10, p.POSITION_CONTROL, target, force=80.0, physicsClientId=self.client
            )
            self._step_sim(1)

    def close_gripper(self):
        force = float(self.policy_params.get("grasp_force", 55.0))
        for _ in range(95):
            p.setJointMotorControl2(
                self.robot_id, 9, p.POSITION_CONTROL, 0.0, force=force, physicsClientId=self.client
            )
            p.setJointMotorControl2(
                self.robot_id, 10, p.POSITION_CONTROL, 0.0, force=force, physicsClientId=self.client
            )
            self._step_sim(1)

    def is_holding_object(self, obj_id: int | None = None) -> bool:
        target = obj_id if obj_id is not None else self.grasped_object
        if target is None:
            return False
        if self.control_mode == "kinematic" and self.constraint_id is not None:
            return True
        left_c = p.getContactPoints(self.robot_id, target, linkIndexA=9, physicsClientId=self.client)
        right_c = p.getContactPoints(self.robot_id, target, linkIndexA=10, physicsClientId=self.client)
        return len(left_c) > 0 and len(right_c) > 0

    def move_to(self, position: list[float]) -> list[float]:
        x = max(-0.35, min(0.40, float(position[0])))
        y = max(-0.32, min(0.35, float(position[1])))
        z = max(0.43, min(0.84, float(position[2])))
        clamped = [x, y, z]

        if not self.policy_params.get("auto_safe_lift", True):
            return self._ik_move(clamped, duration_steps=150)

        current_ee = self.get_ee_pos()
        xy_dist = math.hypot(current_ee[0] - x, current_ee[1] - y)
        safe_z = max(self.get_safe_z(), min(0.82, z + 0.085))
        if xy_dist > 0.035:
            if current_ee[2] < safe_z - 0.005:
                self._ik_move([current_ee[0], current_ee[1], safe_z], duration_steps=110)
            self._ik_move([x, y, safe_z], duration_steps=140)
            return self._ik_move([x, y, z], duration_steps=130)
        return self._ik_move(clamped, duration_steps=140)

    def get_object_position(self, object_name: str) -> list[float] | None:
        return self.vision_locator.locate(object_name)

    def grasp(self, object_name: str) -> bool:
        if object_name not in self.objects:
            return False
        if object_name in ("brown_box", "obstacle_wall"):
            return False

        if self.grasped_object is not None:
            self.release()

        obj_pos = self.vision_locator.locate(object_name)
        if obj_pos is None:
            return False

        obj_id = self.objects[object_name]
        safe_z = self.get_safe_z()
        current_ee = self.get_ee_pos()
        if current_ee[2] < safe_z - 0.03:
            self._ik_move([current_ee[0], current_ee[1], safe_z], duration_steps=100)

        self.open_gripper(slow=False)
        self._ik_move([obj_pos[0], obj_pos[1], safe_z], duration_steps=130)
        self._ik_move([obj_pos[0], obj_pos[1], obj_pos[2] + 0.065], duration_steps=100)
        grasp_z = max(0.438, obj_pos[2] - 0.004)
        self._ik_move([obj_pos[0], obj_pos[1], grasp_z], duration_steps=100)

        self.close_gripper()

        if self.control_mode == "kinematic":
            ee_pos, ee_orn = p.getLinkState(self.robot_id, self.ee_link, physicsClientId=self.client)[:2]
            op, oo = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
            inv_p, inv_o = p.invertTransform(ee_pos, ee_orn)
            rel_p, rel_o = p.multiplyTransforms(inv_p, inv_o, op, oo)
            self.constraint_id = p.createConstraint(
                self.robot_id, self.ee_link, obj_id, -1, p.JOINT_FIXED,
                [0, 0, 0], rel_p, [0, 0, 0], rel_o, [0, 0, 0, 1], physicsClientId=self.client,
            )
            p.changeConstraint(self.constraint_id, maxForce=400, physicsClientId=self.client)

        self.grasped_object = obj_id
        self.grasped_name = object_name

        self._ik_move([obj_pos[0], obj_pos[1], safe_z], duration_steps=130)
        if not self.is_holding_object(obj_id):
            self.grasped_object = None
            self.grasped_name = None
            return False
        return True

    def release(self) -> bool:
        if self.grasped_object is None:
            return False

        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id, physicsClientId=self.client)
            self.constraint_id = None

        self.grasped_object = None
        self.grasped_name = None

        self.open_gripper(slow=True)
        self._step_sim(60)

        ee_pos = self.get_ee_pos()
        if hasattr(self.vision_locator, "clear_cache"):
            self.vision_locator.clear_cache()
        safe_z = self.get_safe_z()
        self._ik_move([ee_pos[0], ee_pos[1], max(ee_pos[2] + 0.10, safe_z)], duration_steps=100)
        return True

    def home(self):
        if self.grasped_object is not None:
            self.release()
        ee = self.get_ee_pos()
        if self.policy_params.get("auto_safe_lift", True):
            if ee[0] > -0.12:
                self._ik_move([ee[0], ee[1], max(0.80, self.get_safe_z())], duration_steps=85)
                cur_j = self._get_joint_positions()
                self._move_joints([cur_j[0], -0.35, 0.0, -1.45, 0.0, 1.65, -0.7854], duration_steps=75)
        elif ee[2] < 0.62 and ee[0] > -0.12:
            self._ik_move([ee[0], ee[1], max(0.66, self.get_safe_z())], duration_steps=85)
        self.open_gripper(slow=False)
        self._move_joints(self.rest_poses, duration_steps=95 if self.policy_params.get("auto_safe_lift", True) else 120)

# env.py — PyBullet manipulation environment (GUI & Headless, hollow container, obstacle barrier)
import math
import random
import time
import pybullet as p
import pybullet_data


class RoboTeachEnv:
    TABLE_Z = 0.42
    CUBE_HALF = 0.025

    def __init__(
        self,
        headless: bool = False,
        seed: int | None = None,
        include_container: bool = True,
        include_obstacle: bool = False,
    ):
        self.headless = headless
        self.seed = seed
        self.rng = random.Random(seed)
        self.include_container = include_container
        self.include_obstacle = include_obstacle

        mode = p.DIRECT if headless else p.GUI
        self.client = p.connect(mode)
        if not headless:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0.15, 0.0, 0.5],
                physicsClientId=self.client,
            )

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=180,
            enableConeFriction=1,
            contactBreakingThreshold=0.001,
            physicsClientId=self.client,
        )

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(self.plane, -1, restitution=0.0, physicsClientId=self.client)

        self.table = self._create_table()
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[-0.40, 0.0, self.TABLE_Z],
            useFixedBase=True,
            physicsClientId=self.client,
        )
        self._configure_robot_dynamics()

        self.objects: dict[str, int] = {}
        self.object_dimensions: dict[str, dict] = {}
        self._load_objects()

        self.initial_positions: dict[int, list[float]] = {}
        self.last_domain_params: dict = {}
        for obj_id in self.objects.values():
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
            self.initial_positions[obj_id] = list(pos)

        for _ in range(180):
            p.stepSimulation(physicsClientId=self.client)

    def _configure_robot_dynamics(self):
        # Realistic elastomeric finger pads for contact-force grasping without weld constraints
        for finger_link in (9, 10):
            p.changeDynamics(
                self.robot_id,
                finger_link,
                lateralFriction=3.0,
                spinningFriction=0.4,
                rollingFriction=0.02,
                frictionAnchor=1,
                restitution=0.0,
                physicsClientId=self.client,
            )
        for j in range(7):
            p.changeDynamics(
                self.robot_id,
                j,
                linearDamping=0.04,
                angularDamping=0.04,
                physicsClientId=self.client,
            )

    def _create_table(self) -> int:
        table_size = [0.55, 0.55, 0.02]
        table_pos = [0.05, 0.0, 0.40]
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=table_size,
            rgbaColor=[0.82, 0.82, 0.84, 1.0],
            physicsClientId=self.client,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=table_size, physicsClientId=self.client
        )
        table = p.createMultiBody(
            0, collision, visual, table_pos, physicsClientId=self.client
        )
        p.changeDynamics(
            table, -1, lateralFriction=1.2, restitution=0.0, physicsClientId=self.client
        )
        return table

    def _apply_cube_dynamics(self, obj_id: int):
        p.changeDynamics(
            obj_id,
            -1,
            lateralFriction=1.6,
            spinningFriction=0.15,
            rollingFriction=0.01,
            linearDamping=0.15,
            angularDamping=0.20,
            frictionAnchor=1,
            restitution=0.0,
            physicsClientId=self.client,
        )

    def _create_hollow_container(self, position: list[float]) -> int:
        # 5-part compound hollow box: 17cm x 17cm outer, 13.8cm x 13.8cm cavity, 7cm wall height
        rgba = [0.58, 0.29, 0.08, 1.0]
        half_list = [
            [0.085, 0.085, 0.005],  # floor
            [0.085, 0.008, 0.035],  # +Y wall
            [0.085, 0.008, 0.035],  # -Y wall
            [0.008, 0.069, 0.035],  # +X wall
            [0.008, 0.069, 0.035],  # -X wall
        ]
        offsets = [
            [0.0, 0.0, 0.0],
            [0.0, 0.077, 0.030],
            [0.0, -0.077, 0.030],
            [0.077, 0.0, 0.030],
            [-0.077, 0.0, 0.030],
        ]
        col_id = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_BOX] * 5,
            halfExtents=half_list,
            collisionFramePositions=offsets,
            physicsClientId=self.client,
        )
        vis_id = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * 5,
            halfExtents=half_list,
            visualFramePositions=offsets,
            rgbaColors=[rgba] * 5,
            physicsClientId=self.client,
        )
        box_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            box_id, -1, lateralFriction=1.2, restitution=0.0, physicsClientId=self.client
        )
        return box_id

    def _load_objects(self):
        h = [self.CUBE_HALF, self.CUBE_HALF, self.CUBE_HALF]
        cube_specs = [
            ("red_cube", [0.95, 0.08, 0.08, 1.0], [0.10, -0.11, 0.445]),
            ("blue_cube", [0.08, 0.18, 0.95, 1.0], [0.10, 0.06, 0.445]),
            ("green_cube", [0.08, 0.80, 0.12, 1.0], [0.20, -0.10, 0.445]),
            ("yellow_cube", [0.95, 0.88, 0.08, 1.0], [0.20, 0.04, 0.445]),
        ]
        for name, rgba, pos in cube_specs:
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=h, rgbaColor=rgba, physicsClientId=self.client
            )
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=h, physicsClientId=self.client
            )
            oid = p.createMultiBody(
                0.12, col, vis, pos, physicsClientId=self.client
            )
            self._apply_cube_dynamics(oid)
            self.objects[name] = oid
            self.object_dimensions[name] = {
                "type": "cube",
                "half_extents": list(h),
                "graspable": True,
            }

        if self.include_container:
            box_pos = [0.25, 0.23, 0.425]
            self.objects["brown_box"] = self._create_hollow_container(box_pos)
            self.object_dimensions["brown_box"] = {
                "type": "container",
                "half_extents": [0.085, 0.085, 0.035],
                "cavity_half": [0.068, 0.068],
                "rim_z": 0.490,
                "graspable": False,
            }

        if self.include_obstacle:
            self.set_obstacle_enabled(True)

    def set_obstacle_enabled(self, enabled: bool, position: list[float] | None = None):
        if enabled and "obstacle_wall" not in self.objects:
            wall_half = [0.11, 0.014, 0.055]  # 22cm along X, 2.8cm thick, 11cm tall (top z = 0.53m)
            wall_pos = position or [0.16, 0.0, 0.475]
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=wall_half,
                rgbaColor=[0.56, 0.12, 0.64, 1.0],
                physicsClientId=self.client,
            )
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=wall_half, physicsClientId=self.client
            )
            wid = p.createMultiBody(
                0.0, col, vis, wall_pos, physicsClientId=self.client
            )
            self.objects["obstacle_wall"] = wid
            self.object_dimensions["obstacle_wall"] = {
                "type": "obstacle",
                "half_extents": wall_half,
                "top_z": wall_pos[2] + wall_half[2],
                "graspable": False,
            }
        elif not enabled and "obstacle_wall" in self.objects:
            p.removeBody(self.objects.pop("obstacle_wall"), physicsClientId=self.client)
            self.object_dimensions.pop("obstacle_wall", None)

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def run(self, seconds: float = 2.0):
        for _ in range(int(seconds * 240)):
            p.stepSimulation(physicsClientId=self.client)
            if not self.headless:
                time.sleep(1.0 / 240.0)

    def reset(self):
        for obj_id, pos in list(self.initial_positions.items()):
            if obj_id in self.objects.values():
                p.resetBasePositionAndOrientation(
                    obj_id, pos, [0, 0, 0, 1], physicsClientId=self.client
                )
                p.resetBaseVelocity(
                    obj_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client
                )
        rest = [-1.2, 0.0, 0.0, -1.5708, 0.0, 1.8675, -0.7854, 0.0, 0.0, 0.04, 0.04, 0.0]
        num_j = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(min(12, num_j)):
            p.resetJointState(self.robot_id, i, rest[i], targetVelocity=0.0, physicsClientId=self.client)
        p.setJointMotorControlArray(
            self.robot_id,
            list(range(7)),
            p.POSITION_CONTROL,
            targetPositions=rest[:7],
            forces=[240.0, 240.0, 240.0, 240.0, 140.0, 140.0, 140.0],
            positionGains=[0.28] * 7,
            velocityGains=[1.05] * 7,
            physicsClientId=self.client,
        )
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, 0.04, force=80.0, physicsClientId=self.client)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, 0.04, force=80.0, physicsClientId=self.client)
        for _ in range(60):
            p.stepSimulation(physicsClientId=self.client)

    def randomize(
        self,
        initial_state: dict | None = None,
        seed: int | None = None,
        domain_randomization: bool = False,
    ):
        rng = random.Random(seed) if seed is not None else self.rng
        stacks = []
        has_obstacle = False
        if isinstance(initial_state, dict):
            stacks = initial_state.get("stacks", [])
            has_obstacle = bool(initial_state.get("has_obstacle", False))

        if has_obstacle and "obstacle_wall" not in self.objects:
            self.set_obstacle_enabled(True)
        elif not has_obstacle and not self.include_obstacle and "obstacle_wall" in self.objects:
            self.set_obstacle_enabled(False)

        # Apply or restore nominal physics & static landmark pose jitter
        table_fric = rng.uniform(1.0, 2.2) if domain_randomization else 1.2
        p.changeDynamics(self.table, -1, lateralFriction=table_fric, restitution=0.0, physicsClientId=self.client)
        cube_masses: dict[str, float] = {}
        cube_frictions: dict[str, float] = {}
        for cname in ("red_cube", "blue_cube", "green_cube", "yellow_cube"):
            oid = self.objects[cname]
            mass = rng.uniform(0.084, 0.156) if domain_randomization else 0.12
            fric = rng.uniform(1.2, 2.8) if domain_randomization else 1.6
            cube_masses[cname] = round(mass, 4)
            cube_frictions[cname] = round(fric, 3)
            p.changeDynamics(oid, -1, mass=mass, lateralFriction=fric, physicsClientId=self.client)

        if "brown_box" in self.objects:
            dx = rng.uniform(-0.008, 0.008) if domain_randomization else 0.0
            dy = rng.uniform(-0.008, 0.008) if domain_randomization else 0.0
            p.resetBasePositionAndOrientation(
                self.objects["brown_box"], [0.25 + dx, 0.23 + dy, 0.425], [0, 0, 0, 1], physicsClientId=self.client
            )
        if "obstacle_wall" in self.objects:
            dx = rng.uniform(-0.008, 0.008) if domain_randomization else 0.0
            dy = rng.uniform(-0.008, 0.008) if domain_randomization else 0.0
            p.resetBasePositionAndOrientation(
                self.objects["obstacle_wall"], [0.16 + dx, 0.0 + dy, 0.475], [0, 0, 0, 1], physicsClientId=self.client
            )

        self.last_domain_params = {
            "domain_randomization": bool(domain_randomization),
            "table_friction": round(table_fric, 3),
            "cube_masses_kg": cube_masses,
            "cube_frictions": cube_frictions,
        }

        top_to_bottom = {pair[0]: pair[1] for pair in stacks if len(pair) == 2}
        all_cubes = ["red_cube", "blue_cube", "green_cube", "yellow_cube"]
        base_cubes = [c for c in all_cubes if c not in top_to_bottom]

        occupied: list[tuple[float, float, float]] = []
        if "brown_box" in self.objects:
            bp, _ = p.getBasePositionAndOrientation(
                self.objects["brown_box"], physicsClientId=self.client
            )
            occupied.append((bp[0], bp[1], 0.14))

        wall_y = None
        if "obstacle_wall" in self.objects:
            wp, _ = p.getBasePositionAndOrientation(
                self.objects["obstacle_wall"], physicsClientId=self.client
            )
            wall_y = wp[1]

        cube_positions: dict[str, list[float]] = {}
        for idx, name in enumerate(base_cubes):
            obj_id = self.objects[name]
            for _ in range(600):
                x = rng.uniform(0.06, 0.22)
                if wall_y is not None:
                    # Partition cubes cleanly on opposite sides of obstacle wall with hand-width Y clearance
                    y = rng.uniform(-0.23, -0.068) if idx % 2 == 0 else rng.uniform(0.066, 0.110)
                else:
                    y = rng.uniform(-0.24, 0.08)
                too_close = False
                for ox, oy, rad in occupied:
                    if math.hypot(x - ox, y - oy) < rad:
                        too_close = True
                        break
                    if wall_y is None and rad < 0.12 and abs(x - ox) < 0.138 and abs(y - oy) < 0.072:
                        too_close = True
                        break
                if too_close:
                    continue
                break
            occupied.append((x, y, 0.088))
            z = self.TABLE_Z + self.CUBE_HALF
            p.resetBasePositionAndOrientation(
                obj_id, [x, y, z], [0, 0, 0, 1], physicsClientId=self.client
            )
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
            cube_positions[name] = [x, y, z]

        placed = set(base_cubes)
        remaining = set(top_to_bottom.keys())
        while remaining:
            progressed = False
            for top_cube in list(remaining):
                bottom = top_to_bottom[top_cube]
                if bottom in placed:
                    bx, by, bz = cube_positions[bottom]
                    tz = bz + 2.0 * self.CUBE_HALF + 0.001
                    oid = self.objects[top_cube]
                    p.resetBasePositionAndOrientation(
                        oid, [bx, by, tz], [0, 0, 0, 1], physicsClientId=self.client
                    )
                    p.resetBaseVelocity(
                        oid, [0, 0, 0], [0, 0, 0], physicsClientId=self.client
                    )
                    cube_positions[top_cube] = [bx, by, tz]
                    placed.add(top_cube)
                    remaining.remove(top_cube)
                    progressed = True
            if not progressed:
                for top_cube in list(remaining):
                    oid = self.objects[top_cube]
                    p.resetBasePositionAndOrientation(
                        oid, [0.10, -0.18, self.TABLE_Z + self.CUBE_HALF], [0, 0, 0, 1],
                        physicsClientId=self.client,
                    )
                break

        for _ in range(90):
            p.stepSimulation(physicsClientId=self.client)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

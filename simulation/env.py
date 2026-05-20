# env.py — PyBullet simülasyon ortamı (Zengin sahne)
import pybullet as p
import pybullet_data
import time
import random


class RoboTeachEnv:
    def __init__(self):
        self.client = p.connect(p.GUI)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.2, cameraYaw=50,
            cameraPitch=-35, cameraTargetPosition=[0, 0, 0.5]
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSolverIterations=100)

        # Zemin
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, restitution=0.0)

        # Masa
        self.table = self._create_table()

        # Robot kolu — Franka Panda
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[-0.4, 0, 0.42],
            useFixedBase=True
        )

        # Nesneler
        self.objects = {}
        self._load_objects()

        # Başlangıç pozisyonlarını sakla
        self.initial_positions = {}
        for name, obj_id in self.objects.items():
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            self.initial_positions[obj_id] = list(pos)

        # Bilinen nesne boyutları (VisionLocator için)
        self.object_dimensions = {
            "red_cube": {"type": "box", "half_extents": [0.025, 0.025, 0.025],
                         "color_rgb": [255, 0, 0]},
            "blue_cube": {"type": "box", "half_extents": [0.025, 0.025, 0.025],
                          "color_rgb": [0, 0, 255]},
            "green_cube": {"type": "box", "half_extents": [0.025, 0.025, 0.025],
                           "color_rgb": [0, 255, 0]},
            "yellow_cube": {"type": "box", "half_extents": [0.025, 0.025, 0.025],
                            "color_rgb": [255, 255, 0]},
            "brown_box": {"type": "box", "half_extents": [0.08, 0.06, 0.04],
                          "color_rgb": [160, 100, 50]},
        }

        # İlk fizik yerleşimi
        for _ in range(240):
            p.stepSimulation()

        print("✅ Simülasyon ortamı hazır!")

    def _create_table(self):
        table_size = [0.5, 0.5, 0.02]
        table_pos = [0, 0, 0.4]
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=table_size,
                                     rgbaColor=[0.6, 0.4, 0.2, 1])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_size)
        table = p.createMultiBody(0, collision, visual, table_pos)
        p.changeDynamics(table, -1, lateralFriction=1.0, restitution=0.0)
        return table

    def _stable_dynamics(self, obj_id):
        """Objeye kararlı fizik parametreleri uygular."""
        p.changeDynamics(obj_id, -1,
                         lateralFriction=1.5,
                         spinningFriction=0.1,
                         rollingFriction=0.01,
                         linearDamping=0.8,
                         angularDamping=0.8,
                         restitution=0.0)

    def _load_objects(self):
        # ─── Renkli küpler (5cm kenar) ─────────────────────
        half = [0.025, 0.025, 0.025]

        # Kırmızı küp
        rv = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[1, 0, 0, 1])
        rc = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        self.objects["red_cube"] = p.createMultiBody(0.2, rc, rv, [0.12, -0.10, 0.45])
        self._stable_dynamics(self.objects["red_cube"])

        # Mavi küp
        bv = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0, 0, 1, 1])
        bc = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        self.objects["blue_cube"] = p.createMultiBody(0.2, bc, bv, [0.12, 0.10, 0.45])
        self._stable_dynamics(self.objects["blue_cube"])

        # Yeşil küp
        gv = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0, 0.85, 0, 1])
        gc = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        self.objects["green_cube"] = p.createMultiBody(0.2, gc, gv, [0.20, -0.05, 0.45])
        self._stable_dynamics(self.objects["green_cube"])

        # Sarı küp
        yv = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[1, 1, 0, 1])
        yc = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        self.objects["yellow_cube"] = p.createMultiBody(0.2, yc, yv, [0.20, 0.05, 0.45])
        self._stable_dynamics(self.objects["yellow_cube"])

        # ─── Açık karton kutu (geniş, 4 küpü rahat alır) ───
        self.objects["brown_box"] = self._create_open_box(
            position=[0.05, 0.0, 0.42],
            inner_size=[0.12, 0.09],    # İç boyut: 24x18cm (4 küp sığar)
            wall_height=0.05,           # Duvar yüksekliği: 5cm
            wall_thickness=0.005,       # Duvar kalınlığı: 5mm
            color=[0.65, 0.45, 0.25, 1] # Karton kahverengisi
        )

        print(f"📦 Nesneler: {list(self.objects.keys())}")

    def _create_open_box(self, position, inner_size, wall_height, wall_thickness, color):
        """Duvarları ve tabanı olan açık bir kutu oluşturur."""
        ix, iy = inner_size
        wh = wall_height
        wt = wall_thickness
        px, py, pz = position

        parts_visual = []
        parts_collision = []
        parts_pos = []

        # Taban
        parts_visual.append(p.createVisualShape(p.GEOM_BOX, halfExtents=[ix, iy, wt], rgbaColor=color))
        parts_collision.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[ix, iy, wt]))
        parts_pos.append([0, 0, 0])

        # Sol duvar (-x)
        parts_visual.append(p.createVisualShape(p.GEOM_BOX, halfExtents=[wt, iy, wh/2], rgbaColor=color))
        parts_collision.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[wt, iy, wh/2]))
        parts_pos.append([-ix, 0, wh/2 + wt])

        # Sağ duvar (+x)
        parts_visual.append(p.createVisualShape(p.GEOM_BOX, halfExtents=[wt, iy, wh/2], rgbaColor=color))
        parts_collision.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[wt, iy, wh/2]))
        parts_pos.append([ix, 0, wh/2 + wt])

        # Arka duvar (-y)
        parts_visual.append(p.createVisualShape(p.GEOM_BOX, halfExtents=[ix, wt, wh/2], rgbaColor=color))
        parts_collision.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[ix, wt, wh/2]))
        parts_pos.append([0, -iy, wh/2 + wt])

        # Ön duvar (+y)
        parts_visual.append(p.createVisualShape(p.GEOM_BOX, halfExtents=[ix, wt, wh/2], rgbaColor=color))
        parts_collision.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[ix, wt, wh/2]))
        parts_pos.append([0, iy, wh/2 + wt])

        # Taban bodysi oluştur
        base_id = p.createMultiBody(
            baseMass=0,  # Sabit (hareket etmeyen)
            baseCollisionShapeIndex=parts_collision[0],
            baseVisualShapeIndex=parts_visual[0],
            basePosition=position,
            linkMasses=[0, 0, 0, 0],
            linkCollisionShapeIndices=parts_collision[1:],
            linkVisualShapeIndices=parts_visual[1:],
            linkPositions=parts_pos[1:],
            linkOrientations=[[0, 0, 0, 1]] * 4,
            linkInertialFramePositions=[[0, 0, 0]] * 4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED] * 4,
            linkJointAxis=[[0, 0, 0]] * 4,
        )

        p.changeDynamics(base_id, -1, lateralFriction=1.0, restitution=0.0)
        for link in range(4):
            p.changeDynamics(base_id, link, lateralFriction=1.0, restitution=0.0)

        return base_id

    def step(self):
        p.stepSimulation()

    def run(self, seconds=5):
        for _ in range(int(seconds * 240)):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def reset(self):
        """Objeleri başlangıç pozisyonlarına geri döndürür."""
        print("  🔄 Sahne sıfırlanıyor...")
        for obj_id, pos in self.initial_positions.items():
            p.resetBasePositionAndOrientation(obj_id, pos, [0, 0, 0, 1])
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
        # Robot eklemlerini sıfırla
        rest = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0, 0, 0, 0.04, 0.04, 0]
        for i in range(12):
            if i < p.getNumJoints(self.robot_id):
                p.resetJointState(self.robot_id, i, rest[i])
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def randomize(self):
        """Küpleri masa üzerinde rastgele yerleştirir (kutu sabit kalır)."""
        print("  🎲 Objeler rastgele yerleştiriliyor...")
        positions = []
        # Kutunun pozisyonunu ekle (küpler kutunun üstüne spawn olmasın)
        box_pos = list(p.getBasePositionAndOrientation(self.objects["brown_box"])[0])
        positions.append((box_pos[0], box_pos[1]))

        movable = ["red_cube", "blue_cube", "green_cube", "yellow_cube"]
        for name in movable:
            obj_id = self.objects[name]
            while True:
                x = random.uniform(0.08, 0.28)
                y = random.uniform(-0.18, 0.18)
                too_close = False
                for px, py in positions:
                    if abs(x - px) < 0.07 and abs(y - py) < 0.07:
                        too_close = True
                        break
                if not too_close:
                    break
            positions.append((x, y))
            p.resetBasePositionAndOrientation(obj_id, [x, y, 0.45], [0, 0, 0, 1])
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        for _ in range(100):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect()
        print("🔴 Simülasyon kapatıldı.")

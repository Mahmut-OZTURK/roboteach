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
        p.setPhysicsEngineParameter(numSolverIterations=150)

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
        p.changeDynamics(table, -1, lateralFriction=2.0, restitution=0.0)
        return table

    def _stable_dynamics(self, obj_id):
        """Objeye kararlı fizik parametreleri uygular."""
        p.changeDynamics(obj_id, -1,
                         lateralFriction=2.0,
                         spinningFriction=0.2,
                         rollingFriction=0.01,
                         linearDamping=0.9,
                         angularDamping=0.9,
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

        print(f"📦 Nesneler: {list(self.objects.keys())}")

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
        rest = [-1.2, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0, 0, 0, 0.04, 0.04, 0]
        for i in range(12):
            if i < p.getNumJoints(self.robot_id):
                p.resetJointState(self.robot_id, i, rest[i])
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def randomize(self, initial_state=None):
        """Küpleri masa üzerinde rastgele yerleştirir (gerekirse başlangıç yığınlarını kurar)."""
        print("  🎲 Objeler yerleştiriliyor...")
        
        stacks = []
        if initial_state and isinstance(initial_state, dict):
            stacks = initial_state.get("stacks", [])
        
        # Hangi küpün hangi küpün üzerinde olduğunu haritalandır
        # stacks: [["top_cube", "bottom_cube"]]
        top_to_bottom = {pair[0]: pair[1] for pair in stacks if len(pair) == 2}
        bottom_to_top = {pair[1]: pair[0] for pair in stacks if len(pair) == 2}
        
        # Sadece en alttaki küpleri veya yığınlanmamış küpleri konumlandıracağız
        all_cubes = ["red_cube", "blue_cube", "green_cube", "yellow_cube"]
        base_cubes = [cube for cube in all_cubes if cube not in top_to_bottom]
        
        positions = []
        cube_positions = {}
        
        # En alttaki veya bağımsız küpleri konumlandır
        for name in base_cubes:
            obj_id = self.objects[name]
            while True:
                x = random.uniform(0.08, 0.28)
                y = random.uniform(-0.18, 0.18)
                too_close = False
                for px, py in positions:
                    if abs(x - px) < 0.08 and abs(y - py) < 0.08:
                        too_close = True
                        break
                if not too_close:
                    break
            positions.append((x, y))
            z = 0.425
            p.resetBasePositionAndOrientation(obj_id, [x, y, z], [0, 0, 0, 1])
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
            cube_positions[name] = [x, y, z]

        # Şimdi üstteki küpleri yığın sırasına göre yerleştir
        placed = set(base_cubes)
        remaining = set(top_to_bottom.keys())
        
        while remaining:
            progress = False
            for top_cube in list(remaining):
                bottom_cube = top_to_bottom[top_cube]
                if bottom_cube in placed:
                    bx, by, bz = cube_positions[bottom_cube]
                    # Üstteki küpü hafifçe boşlukla üzerine koy (küp boyutu 5cm, 2mm tolerans ekliyoruz ki fizik motoru çakışıp fırlatmasın)
                    tz = bz + 0.052
                    obj_id = self.objects[top_cube]
                    p.resetBasePositionAndOrientation(obj_id, [bx, by, tz], [0, 0, 0, 1])
                    p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
                    cube_positions[top_cube] = [bx, by, tz]
                    placed.add(top_cube)
                    remaining.remove(top_cube)
                    progress = True
            if not progress:
                # Kısır döngü durumunda geri kalanları rastgele at
                for top_cube in remaining:
                    obj_id = self.objects[top_cube]
                    while True:
                        x = random.uniform(0.08, 0.28)
                        y = random.uniform(-0.18, 0.18)
                        too_close = False
                        for px, py in positions:
                            if abs(x - px) < 0.08 and abs(y - py) < 0.08:
                                too_close = True
                                break
                        if not too_close:
                            break
                    positions.append((x, y))
                    p.resetBasePositionAndOrientation(obj_id, [x, y, 0.425], [0, 0, 0, 1])
                    p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
                break

        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect()
        print("🔴 Simülasyon kapatıldı.")

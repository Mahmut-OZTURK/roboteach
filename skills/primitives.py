# primitives.py — Robot becerileri (Vision-based, Franka Panda)
# Hareket: Waypoint interpolasyonu + resetJointState (deterministik, kesin)
# Fizik: Sadece obje etkileşimleri için çalışır (yerçekimi, temas, sürtünme)
import pybullet as p
import math
import time


class RobotSkills:
    def __init__(self, robot_id, objects: dict, vision_locator, end_effector_index=11):
        self.robot_id = robot_id
        self.objects = objects
        self.vision_locator = vision_locator
        self.ee_link = end_effector_index  # panda_grasptarget
        self.num_joints = 7
        self.grasped_object = None
        self.constraint_id = None

        self.lower_limits = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        self.upper_limits = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        self.joint_ranges = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9096, 5.9342]
        self.rest_poses = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0]
        self.grip_orn = p.getQuaternionFromEuler([0, math.pi, 0])

    def _solve_ik(self, position):
        """IK çözümünü hesapla (link 11 = panda_grasptarget)."""
        return p.calculateInverseKinematics(
            self.robot_id, self.ee_link, position, self.grip_orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_poses,
            maxNumIterations=1000,
            residualThreshold=1e-5
        )

    def _get_joint_positions(self):
        """Mevcut eklem açılarını oku."""
        return [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]

    def _ik_move(self, target_pos, num_waypoints=50, settle_steps=5):
        """
        Kademeli IK hareketi — kesin ve deterministik.
        Eklem açılarını interpolasyon ile hedef IK çözümüne taşır.
        Her waypoint'te fizik simülasyonu çalışır (constraint + yerçekimi).
        """
        target_angles = self._solve_ik(target_pos)
        current_angles = self._get_joint_positions()

        for wp in range(1, num_waypoints + 1):
            t = wp / num_waypoints
            for i in range(self.num_joints):
                interp_angle = current_angles[i] + (target_angles[i] - current_angles[i]) * t
                p.resetJointState(self.robot_id, i, interp_angle)

            # Her waypoint'te fizik adımları çalıştır (constraint kuvvetleri, yerçekimi)
            for _ in range(settle_steps):
                p.stepSimulation()
            time.sleep(0.002)  # Görsel akıcılık

        # Son pozisyonu doğrula
        final_pos = self.get_ee_pos()
        dist = sum((final_pos[i] - target_pos[i]) ** 2 for i in range(3)) ** 0.5
        if dist < 0.02:
            print(f"     ✅ Ulaşıldı: {[round(x, 3) for x in final_pos]}")
        else:
            print(f"     ⚠️  Hedefe tam ulaşılamadı (hata: {dist:.3f}m)")
        return final_pos

    def get_ee_pos(self):
        """End-effector (panda_grasptarget) pozisyonu."""
        state = p.getLinkState(self.robot_id, self.ee_link)
        return list(state[0])

    def open_gripper(self, slow=False):
        """Kıskacı açar."""
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, 0.04, force=100)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, 0.04, force=100)
        steps = 120 if slow else 60
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def close_gripper(self):
        """Kıskacı kapatır."""
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, 0.0, force=200)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, 0.0, force=200)
        for _ in range(80):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def move_to(self, position, speed=0.5):
        """Kartezyen koordinata git."""
        x = max(-0.4, min(0.4, position[0]))
        y = max(-0.3, min(0.4, position[1]))
        z = max(0.42, min(0.85, position[2]))
        clamped = [x, y, z]
        print(f"  🔵 move_to({[round(v, 3) for v in clamped]})")
        return self._ik_move(clamped)

    def get_object_position(self, object_name):
        """Kameradan nesne konumu al (pure vision)."""
        return self.vision_locator.locate(object_name)

    def grasp(self, object_name):
        """Nesneyi tut."""
        if object_name not in self.objects:
            print(f"  ❌ '{object_name}' bulunamadı!")
            return False

        if self.grasped_object is not None:
            print("  ⚠️  Zaten tutuyorum, bırakıyorum...")
            self.release()

        obj_pos = self.vision_locator.locate(object_name)
        if obj_pos is None:
            print(f"  ❌ '{object_name}' kamerada görülemiyor!")
            return False

        print(f"  🟢 grasp('{object_name}') → {[round(x, 3) for x in obj_pos]}")

        self.open_gripper()

        # Waypoint → yaklaşma → iniş
        self._ik_move([obj_pos[0], obj_pos[1], 0.65])
        self._ik_move([obj_pos[0], obj_pos[1], obj_pos[2] + 0.06])

        grasp_z = obj_pos[2] - 0.02
        grasp_z = max(0.43, grasp_z)
        self._ik_move([obj_pos[0], obj_pos[1], grasp_z])

        self.close_gripper()

        # Constraint — link 11 (grasptarget)
        obj_id = self.objects[object_name]
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(self.constraint_id, maxForce=500)
        self.grasped_object = obj_id
        print(f"     ✅ '{object_name}' tutuldu!")

        # Kaldır
        self._ik_move([obj_pos[0], obj_pos[1], obj_pos[2] + 0.12])
        return True

    def release(self):
        """Tutulan nesneyi bırak."""
        if self.grasped_object is None:
            print("  ⚠️  Tutulmuş nesne yok!")
            return False

        obj_id = self.grasped_object

        # 1. Constraint kaldır + momentum sıfırla
        p.removeConstraint(self.constraint_id)
        self.constraint_id = None
        self.grasped_object = None
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        # 2. Kıskacı yavaşça aç
        self.open_gripper(slow=True)

        # 3. Yerleşmeyi bekle
        for _ in range(300):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        print("     ✅ Nesne bırakıldı!")

        # 4. Yukarı çekil
        ee_pos = self.get_ee_pos()
        self._ik_move([ee_pos[0], ee_pos[1], ee_pos[2] + 0.15])
        return True

    def home(self):
        """Başlangıç pozisyonuna dön."""
        print("  🏠 home()")
        if self.grasped_object is not None:
            self.release()

        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, self.rest_poses[i])
        # Parmakları aç
        p.resetJointState(self.robot_id, 9, 0.04)
        p.resetJointState(self.robot_id, 10, 0.04)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        print("     ✅ Eve döndü!")

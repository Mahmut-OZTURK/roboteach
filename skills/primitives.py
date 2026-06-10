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
        self.rest_poses = [-1.2, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0]
        self.grip_orn = p.getQuaternionFromEuler([0, math.pi, 0])

    def get_safe_z(self):
        """Masadaki tüm küplerin Z pozisyonunu sorgulayıp en yüksek olanına göre dinamik güvenli geçiş Z seviyesi döner."""
        max_z = 0.425
        for obj_id in self.objects.values():
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if pos[2] > max_z:
                max_z = pos[2]
        # Küpler 0.05m boyunda. En yüksek küpün üstünden 0.12m yukarısı güvenli yörüngedir.
        return max(0.65, max_z + 0.12)

    def _stabilize_cubes(self):
        """Tutulmayan tüm küplerin hız ve açısal momentumunu sıfırlar, yönelimlerini dikleştirir.
        Bu, _ik_move ve home() sırasında çalışan yüzlerce stepSimulation'ın
        birikimli sayısal hatasını önleyerek kulelerin devrilmesini engeller."""
        for obj_id in self.objects.values():
            if obj_id == self.grasped_object:
                continue
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            p.resetBasePositionAndOrientation(obj_id, pos, [0, 0, 0, 1])
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

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

    def _ik_move(self, target_pos, num_waypoints=95, settle_steps=3):
        """
        Kademeli IK hareketi — son derece yavaş, yumuşak ve kararlı.
        """
        target_angles = self._solve_ik(target_pos)
        current_angles = self._get_joint_positions()

        for wp in range(1, num_waypoints + 1):
            t = wp / num_waypoints
            for i in range(self.num_joints):
                interp_angle = current_angles[i] + (target_angles[i] - current_angles[i]) * t
                p.resetJointState(self.robot_id, i, interp_angle)

            # Fizik adımlarını çalıştır
            for _ in range(settle_steps):
                p.stepSimulation()
            time.sleep(0.012)  # Ağır ve kararlı hareket için hızı yavaşlattık

        # Son pozisyonu doğrula
        final_pos = self.get_ee_pos()
        dist = sum((final_pos[i] - target_pos[i]) ** 2 for i in range(3)) ** 0.5
        if dist < 0.02:
            print(f"     ✅ Ulaşıldı: {[round(x, 3) for x in final_pos]}")
        else:
            print(f"     ⚠️  Hedefe tam ulaşılamadı (hata: {dist:.3f}m)")

        # Hareket sonrası tüm serbest küpleri stabilize et (drift önleme)
        self._stabilize_cubes()

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
        """Kartezyen koordinata git. Çarpışmaları önlemek için akıllı yükselme mekanizması içerir."""
        x = max(-0.4, min(0.4, position[0]))
        y = max(-0.3, min(0.4, position[1]))
        z = max(0.42, min(0.85, position[2]))
        clamped = [x, y, z]
        
        current_ee = self.get_ee_pos()
        xy_dist = math.sqrt((current_ee[0] - x)**2 + (current_ee[1] - y)**2)
        safe_z = self.get_safe_z()
        
        # Eğer XY düzleminde hareket varsa ve Z seviyelerinden biri güvenli Z'nin altındaysa,
        # çarpışmayı önlemek için hareketi 3 aşamalı (yukarı -> yatay -> aşağı) yap.
        if xy_dist > 0.02 and (current_ee[2] < safe_z - 0.01 or z < safe_z - 0.01):
            print(f"     🛡️ Güvenli Hareket: Dikey kalkış, yatay taşıma ve dikey iniş uygulanıyor (safe_z: {safe_z:.3f}m)")
            # 1. Mevcut XY koordinatında güvenli yüksekliğe çık (eğer zaten yüksekte değilse)
            if current_ee[2] < safe_z:
                self._ik_move([current_ee[0], current_ee[1], safe_z])
            # 2. Güvenli yükseklikte hedef XY koordinatına git
            self._ik_move([x, y, safe_z])
            # 3. Hedef XY koordinatında hedef Z yüksekliğine in
            return self._ik_move([x, y, z])
        else:
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

        # Robot parmaklarının mevcut yerinden yeni küpe yatayda giderken masadakileri devirmemesi için:
        # Önce mevcut konumunda Z ekseninde güvenli yüksekliğe kalk, sonra yeni küpün XY'sine git.
        current_ee = self.get_ee_pos()
        safe_z = self.get_safe_z()
        if current_ee[2] < safe_z - 0.05:
            print(f"     ⬆️ Grasp öncesi güvenli yüksekliğe kalkılıyor: {safe_z:.3f}m")
            self._ik_move([current_ee[0], current_ee[1], safe_z])

        print(f"  🟢 grasp('{object_name}') → {[round(x, 3) for x in obj_pos]}")

        obj_id = self.objects[object_name]

        self.open_gripper()

        # 1. Yüksekten yaklaş
        safe_z = self.get_safe_z()
        self._ik_move([obj_pos[0], obj_pos[1], safe_z])

        # 2. Kademeli iniş — objenin tam üzerinde hizalan
        self._ik_move([obj_pos[0], obj_pos[1], obj_pos[2] + 0.06])

        # 3. Son iniş — parmakları küpün tam merkezine getir
        grasp_z = obj_pos[2]
        grasp_z = max(0.43, grasp_z)
        self._ik_move([obj_pos[0], obj_pos[1], grasp_z])

        self.close_gripper()

        # 5. Kusursuz diklik hizalaması ve sıfır momentum
        # Küpü mükemmel şekilde dikleştirerek kulelerin yamulmasını önler
        obj_pos_now, _ = p.getBasePositionAndOrientation(obj_id)
        p.resetBasePositionAndOrientation(obj_id, obj_pos_now, [0, 0, 0, 1])
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        # 6. Constraint — göreceli dönüşüm ile (uzaylı gibi zıplamasını önler)
        ee_state = p.getLinkState(self.robot_id, self.ee_link)
        ee_pos, ee_orn = ee_state[0], ee_state[1]
        obj_pos_now, obj_orn_now = p.getBasePositionAndOrientation(obj_id)

        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, obj_pos_now, obj_orn_now)

        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn,
            childFrameOrientation=[0, 0, 0, 1]
        )
        p.changeConstraint(self.constraint_id, maxForce=500)
        self.grasped_object = obj_id
        
        # Sadece kavranan nesne ile robot arasındaki çarpışmaları kapat (Constraint çatışmasını önlemek için)
        for link_idx in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(self.robot_id, obj_id, link_idx, -1, enableCollision=0)

        print(f"     ✅ '{object_name}' tutuldu!")

        # 7. Kaldır
        safe_z = self.get_safe_z()
        self._ik_move([obj_pos[0], obj_pos[1], safe_z])
        return True

    def release(self):
        """Tutulan nesneyi bırak — yüzeye hafifçe temas ettikten sonra bırak."""
        if self.grasped_object is None:
            print("  ⚠️  Tutulmuş nesne yok!")
            return False

        obj_id = self.grasped_object

        # 1. Momentumu sıfırla (nesne duraganlığı)
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        # 2. Constraint kaldır
        p.removeConstraint(self.constraint_id)
        self.constraint_id = None
        self.grasped_object = None

        # 3. Momentum tekrar sıfırla (constraint kaldırılınca oluşan tepki kuvvetini öldür)
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        # 4. Kıskacı yavaşça aç
        self.open_gripper(slow=True)

        # 5. Doğal settle süresi (yerçekimi ve temas fizikleri doğal yerleşsin)
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        # 6. Post-settle stabilizasyon: Bırakılan küpü dikleştir ve tüm küplerin birikmiş driftini sıfırla
        obj_pos_final, _ = p.getBasePositionAndOrientation(obj_id)
        p.resetBasePositionAndOrientation(obj_id, obj_pos_final, [0, 0, 0, 1])
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
        self._stabilize_cubes()

        print("     ✅ Nesne bırakıldı!")

        # 7. Yukarı çekil (Dikey olarak)
        ee_pos = self.get_ee_pos()
        safe_z = self.get_safe_z()
        target_lift_z = max(ee_pos[2] + 0.15, safe_z)
        self._ik_move([ee_pos[0], ee_pos[1], target_lift_z])

        # 8. Robot geri çekildikten sonra bırakılan nesnenin robotla olan çarpışmasını geri etkinleştir
        for link_idx in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(self.robot_id, obj_id, link_idx, -1, enableCollision=1)

        return True

    def home(self):
        """Başlangıç pozisyonuna dön (Yavaş ve son derece pürüzsüz enterpolasyon)."""
        print("  🏠 home()")
        if self.grasped_object is not None:
            self.release()

        current_angles = self._get_joint_positions()
        num_steps = 80
        
        for wp in range(1, num_steps + 1):
            t = wp / num_steps
            for i in range(self.num_joints):
                interp_angle = current_angles[i] + (self.rest_poses[i] - current_angles[i]) * t
                p.resetJointState(self.robot_id, i, interp_angle)
            # Parmakları yavaşça aç
            finger_val = 0.04 * t
            p.resetJointState(self.robot_id, 9, finger_val)
            p.resetJointState(self.robot_id, 10, finger_val)
            
            p.stepSimulation()
            time.sleep(0.012)
            
        print("     ✅ Eve döndü!")

        # Eve döndükten sonra tüm küpleri stabilize et
        self._stabilize_cubes()

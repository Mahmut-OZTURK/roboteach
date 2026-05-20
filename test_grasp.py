import pybullet as p, pybullet_data, time, math
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.15, -0.08, 0.5])

# table
col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=[0.6, 0.4, 0.2, 1])
p.createMultiBody(0, col, vis, [0, 0, 0.4])

# cube
rc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
rv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[1, 0, 0, 1])
cube = p.createMultiBody(0.1, rc, rv, [0.15, -0.08, 0.45])

robot = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[-0.3, 0, 0.42], useFixedBase=True)

# Reach
pos = [0.15, -0.08, 0.50]
orn = p.getQuaternionFromEuler([0, math.pi, 0])
rest = [0, 0.4, 0, -1.5, 0, 1.2, 0]
angles = p.calculateInverseKinematics(robot, 6, pos, orn, restPoses=rest)
for i in range(7): p.resetJointState(robot, i, angles[i])
for _ in range(50): p.stepSimulation()

# Constraint at [0,0,0.05]
c = p.createConstraint(robot, 6, cube, -1, p.JOINT_FIXED, [0,0,0], [0,0,0.05], [0,0,0])
p.changeConstraint(c, maxForce=500)

# Lift
pos = [0.15, -0.08, 0.70]
for _ in range(100):
    angles = p.calculateInverseKinematics(robot, 6, pos, orn, restPoses=rest)
    for i in range(7): p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, angles[i], force=500)
    p.stepSimulation()
    time.sleep(1/240)

time.sleep(2)
p.disconnect()

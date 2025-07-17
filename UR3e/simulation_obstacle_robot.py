import os
import math 
import numpy as np
import time
import pybullet 
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from obstacle_scene import ObstacleScene

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur3e.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")

class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]


class UR3eSim():
  
    def __init__(self, camera_attached=False):
        pybullet.connect(pybullet.GUI)
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setRealTimeSimulation(True)
        
        self.end_effector_index = 7
        self.ur3e = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur3e)
        

        self.obstacles = []
        
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur3e, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur3e, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info     


    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        # table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
        robot = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        return robot
        

    

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur3e, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )


    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur3e, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur3e, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       

    def add_gui_sliders(self, initial_pose=[0.05, 0.1, 0.1, 0, 0, 0]):
        self.sliders = []
        self.sliders.append(pybullet.addUserDebugParameter("X", -0.4, 0.4, initial_pose[0]))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -0.4, 0.4, initial_pose[1]))
        self.sliders.append(pybullet.addUserDebugParameter("Z", -0.4, 0.4, initial_pose[2]))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, initial_pose[3]))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, initial_pose[4]))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, initial_pose[5]))


    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        Rx = pybullet.readUserDebugParameter(self.sliders[3])
        Ry = pybullet.readUserDebugParameter(self.sliders[4])
        Rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur3e, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

    def remove_obstacles(self):
        """Remove all obstacles from the scene."""
        for obstacle_id in self.obstacles:
            pybullet.removeBody(obstacle_id)
        self.obstacles.clear()
        print("All obstacles have been removed.")


def demo_with_obstacles(obstacle_type="simple"):

    sim = UR3eSim()
    sim.obstacles = ObstacleScene(scene_type=obstacle_type).create_obstacle_scene()
    sim.add_gui_sliders()
    while True:
        x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
        joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
        sim.set_joint_angles(joint_angles)
        if sim.check_collisions():
            print("Collision detected!")
        time.sleep(0.01)

def demo_simulation():
    """ Demo program showing how to use the sim """ 
    sim = UR3eSim()
    sim.add_gui_sliders()
    while True:
        x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
        joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
        sim.set_joint_angles(joint_angles)
        sim.check_collisions()



if __name__ == "__main__":
    print("Choose a simulation mode:")
    print("1. Basic UR3e simulation")
    print("2. Simple obstacle scene")
    print("3. Complex obstacle scene")
    choice = input("Enter 3: ").strip()
    if choice == "1":
        demo_simulation()
    elif choice == "2":
        demo_with_obstacles(obstacle_type="simple")
    elif choice == "3":
        demo_with_obstacles(obstacle_type="complex")
    else:
        print("Invalid choice. Exiting.")
        exit(1)
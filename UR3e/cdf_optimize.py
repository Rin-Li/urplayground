from ur_sdf.cdf.nn_cdf import CDF
from ur_sdf.cdf.mlp import MLPRegression
import pybullet
import torch
import torch.nn as nn
import numpy as np
import os
import pybullet_data
from collections import namedtuple
import casadi as ca
import time

ROBOT_URDF_PATH = "UR3e/ur_e_description/urdf/ur3e.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")

class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]

class UR3Sim():
    def __init__(self, camera_attached=False):
        pybullet.connect(pybullet.GUI)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setRealTimeSimulation(True)

        self.end_effector_index = 7
        self.ur3e = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur3e)

        self.control_joints = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
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

    def create_obstacle(self, position, radius):
        col_id = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=radius)
        vis_id = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
        pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position.tolist()
        )

    def create_ring(self, radius, center, rot, thickness=0.05):
        theta = np.arange(0, 2*np.pi, 0.2)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)
        points = np.stack([x, y, z], axis=-1)
        points = points @ rot.T + center
        for pt in points:
            self.create_obstacle(pt, thickness)

    def create_wall(self, size, center, rot, thickness=0.05):
        x = np.arange(-size[0]/2, size[0]/2, 0.05)
        y = np.arange(-size[1]/2, size[1]/2, 0.05)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)
        points = np.stack([x, y, z], axis=-1)
        points = points @ rot.T + center
        for pt in points:
            self.create_obstacle(pt, thickness)

def create_system_matrices(n, dt):
    A_d = np.eye(n)
    B_d = np.eye(n) * dt
    return A_d, B_d

def solve_optimization_problem(n, x0, xf, cons_u, A, B, distance, gradient, dt, solver, safety_buffer=0.3):
    X = ca.MX.sym('X', n, 2)
    U = ca.MX.sym('U', n, 1)
    Q = np.diag([150, 190, 80, 70, 70, 90])
    R = np.diag([0.01]*n)

    obj = ca.mtimes((X[:,1]-xf).T, Q @ (X[:,1]-xf)) + ca.mtimes(U.T, R @ U)

    g = []
    g.append(X[:,0] - x0)
    g.append(X[:,1] - (A @ X[:,0] + B @ U[:,0]))
    g.append(-ca.mtimes(gradient, U) * dt - np.log(distance + safety_buffer))

    lbg = [0]*n*2 + [-np.inf]
    ubg = [0]*n*2 + [0]

    lbx = ca.vertcat(*([[-10]*n]*2 + [[-cons_u]*n]))
    ubx = ca.vertcat(*([[10]*n]*2 + [[cons_u]*n]))

    x_vec = ca.vertcat(ca.reshape(X, n*2, 1), ca.reshape(U, n, 1))
    g_vec = ca.vertcat(*g)

    qp = {'x': x_vec, 'f': obj, 'g': g_vec}
    solver = ca.nlpsol('solver', solver, qp, {'print_time': 0, 'verbose': False})
    sol = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    opt_x = np.array(sol['x'][:n*2].full()).reshape(2, n).T
    opt_u = np.array(sol['x'][n*2:].full()).reshape(1, n).T
    return opt_x, opt_u

def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdf = CDF(device)
    model = MLPRegression(input_dims=9, output_dims=1, mlp_layers=[1024,512,256,128,128], skips=[], act_fn=nn.ReLU, nerf=True)
    model.load_state_dict(torch.load("ur_sdf/cdf/model_dict.pt")[19900])
    model.to(device)

    dt = 0.01
    N = 500
    A, B = create_system_matrices(6, dt)
    x0 = np.array([2.474, -0.419, 2.271, -2.783, 2.217, 2.459])
    xf = np.array([-1.331, -3.275, 1.957, -3.239, -2.527, 3.285])
    solver = 'ipopt'

    log_x = []
    log_u = []
    log_d = []

    for i in range(N):
        log_x.append(x0)

        # Build obstacles
        wall_center = torch.tensor([0.5, 0.0, 0.2], device=device, dtype=torch.float32)
        wall_rot = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]], device=device, dtype=torch.float32)
        ring_center = torch.tensor([0.3, 0.0, 0.45], device=device, dtype=torch.float32)
        ring_rot = torch.tensor([[0,0,-1],[0,1,0],[1,0,0]], device=device, dtype=torch.float32)

        def ring(radius, center, rot):
            theta = torch.arange(0, 2*3.14, 0.2, device=device)
            x = radius * torch.cos(theta)
            y = radius * torch.sin(theta)
            z = torch.zeros_like(x)
            pts = torch.stack([x,y,z], dim=-1)
            return pts @ rot.T + center

        def wall(size, center, rot):
            x = torch.arange(-size[0]/2, size[0]/2, 0.05, device=device)
            y = torch.arange(-size[1]/2, size[1]/2, 0.05, device=device)
            x, y = torch.meshgrid(x, y)
            pts = torch.stack([x.flatten(), y.flatten(), torch.zeros_like(x.flatten())], dim=-1)
            return pts @ rot.T + center

        pts = torch.cat([wall(torch.tensor([0.5,0.5],device=device), wall_center, wall_rot),
                         ring(0.4, ring_center, ring_rot)], dim=0)

        x0_t = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_()
        dist, grad = cdf.inference_d_wrt_q(pts, x0_t, model, return_grad=True)
        log_d.append(dist.item())
        dist = dist.detach().cpu().numpy()
        grad = grad.detach().cpu().numpy()

        opt_x, opt_u = solve_optimization_problem(6, x0, xf, 2.7, A, B, dist, grad, dt, solver, 0.3)
        x0 = A @ opt_x[:, 0] + B @ opt_u[:, 0]
        log_u.append(opt_u[:, 0])

        if i > 2 and np.linalg.norm(opt_u - log_u[-2]) < 0.01:
            print(f"Stopped at step {i}")
            break

    UR3e = UR3Sim()
    UR3e.create_wall([0.5, 0.5], [0.5, 0.0, 0.2], np.array([[1,0,0],[0,0,-1],[0,1,0]]))
    UR3e.create_ring(0.4, [0.3, 0.0, 0.45], np.array([[0,0,-1],[0,1,0],[1,0,0]]))

    def playback():
        for x in log_x:
            UR3e.set_joint_angles(x)
            print("Current Joint Angles: ", x)
            pybullet.stepSimulation()
            time.sleep(0.01)  

    print("Initial trajectory playback...")
    playback()

    print("Trajectory complete. Press Enter to replay, Ctrl+C to quit.")
    while True:
        try:
            input(">>> Press Enter to replay the motion... ")
            playback()
        except KeyboardInterrupt:
            print("Exiting replay loop.")
            break

if __name__ == "__main__":
    main()

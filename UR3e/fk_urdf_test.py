import pybullet as p, pybullet_data, time, os, torch, numpy as np
from ur_sdf.ur3e.ur3e import URRobot


MESH_DIR = '/home/kklab-ur-robot/ur_sdf/ur3e/model'


URDF_PATH = '/home/kklab-ur-robot/urplayground/UR3e/ur_e_description/urdf/ur3e.urdf'

device   = 'cpu'
ur_model = URRobot(device=device)


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')


vis_ids = []
for link in ur_model.meshes.keys():
    stl = os.path.join(MESH_DIR, f'{link}.stl')
    vs  = p.createVisualShape(p.GEOM_MESH, fileName=stl,
                              meshScale=[1,1,1], rgbaColor=[0.6,0.6,0.9,1])
    mb  = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs)
    vis_ids.append(mb)


official_id = p.loadURDF(
    URDF_PATH,
    basePosition=[0.8, 0, 0],  
    useFixedBase=True
)


def mat2quat(R):
    t = np.trace(R)
    if t>0:
        s = np.sqrt(t+1.0)*2
        qw = 0.25*s
        qx = (R[2,1]-R[1,2])/s
        qy = (R[0,2]-R[2,0])/s
        qz = (R[1,0]-R[0,1])/s
    else:
        i = np.argmax([R[0,0],R[1,1],R[2,2]])
        if i==0:
            s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            qw = (R[2,1]-R[1,2])/s
            qx = 0.25*s
            qy = (R[0,1]+R[1,0])/s
            qz = (R[0,2]+R[2,0])/s
        elif i==1:
            s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
            qw = (R[0,2]-R[2,0])/s
            qx = (R[0,1]+R[1,0])/s
            qy = 0.25*s
            qz = (R[1,2]+R[2,1])/s
        else:
            s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            qw = (R[1,0]-R[0,1])/s
            qx = (R[0,2]+R[2,0])/s
            qy = (R[1,2]+R[2,1])/s
            qz = 0.25*s
    return [qx,qy,qz,qw]


traj = [
    np.zeros(6),
    np.array([0.6,-1.1,1.4,-1.8,0.8,0.3]),
    np.random.uniform(-2,2,6)
]

for q in traj:
    q_t  = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    T_ws = ur_model.get_transformations_each_link(torch.eye(4).unsqueeze(0), q_t)


    for i, T in enumerate(T_ws):
        pos = T[0,:3,3].numpy()
        orn = mat2quat(T[0,:3,:3].numpy())
        p.resetBasePositionAndOrientation(vis_ids[i], pos, orn)


    for j in range(6):
        p.resetJointState(official_id, j, q[j])


    for _ in range(240):
        p.stepSimulation()
        time.sleep(1/240)


while p.isConnected():
    p.stepSimulation()
    time.sleep(1/240)

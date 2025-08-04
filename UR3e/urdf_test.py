import pybullet as p, pybullet_data

def load_mesh_as_body(path, rgba, scale=[1, 1, 1]):

    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=scale)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=path,
                                 meshScale=scale, rgbaColor=rgba)
    return p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=col_id,
                             baseVisualShapeIndex=vis_id)

# -----------------------------------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", useFixedBase=True)


load_mesh_as_body("/home/kklab-ur-robot/urplayground/ur_sdf/ur3e/model/shoulder.stl", rgba=[1, 0, 0, 0.5])


load_mesh_as_body("/home/kklab-ur-robot/urplayground/UR3e/ur_e_description/meshes/ur3e/collision/shoulder.stl",
                  rgba=[0, 0, 1, 0.5])

print("按 ESC 退出 PyBullet 窗口")
while p.isConnected():
    p.stepSimulation()

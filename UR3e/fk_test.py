import pybullet as p, pybullet_data, numpy as np, torch
from UR3e.UR3eSim import UR3eSim
from ur_sdf.ur3e.ur3e import URRobot

device   = 'cpu'
ur_model = URRobot(device=device)
sim      = UR3eSim()


my_names   = ['base','shoulder','upperarm','forearm','wrist1','wrist2','wrist3']
pb_names   = ['base_link','shoulder_link','upper_arm_link',
              'forearm_link','wrist_1_link','wrist_2_link','wrist_3_link']
name_map   = dict(zip(my_names, pb_names))


child2idx  = { p.getJointInfo(sim.ur3e,i)[12].decode(): i
               for i in range(p.getNumJoints(sim.ur3e)) }
child2idx['base_link'] = -1   

def fk_pybullet(q):
    sim.set_joint_angles(q)
    outs = {}
    for my, pb in name_map.items():
        if child2idx[pb] == -1:      # base
            pos, orn = p.getBasePositionAndOrientation(sim.ur3e)
        else:
            pos, orn = p.getLinkState(sim.ur3e, child2idx[pb],
                                      computeForwardKinematics=True)[:2]
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        outs[my] = (np.asarray(pos), R)
    return outs

def fk_ours(q):
    q_t   = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    T_all = ur_model.get_transformations_each_link(
              torch.eye(4).unsqueeze(0), q_t)
    return {name: (T_all[i][0,:3,3].numpy(), T_all[i][0,:3,:3].numpy())
            for i, name in enumerate(my_names)}


for q in [np.zeros(6),
          np.array([ 0.6,-1.1, 1.4,-1.8, 0.8, 0.3]),
          np.random.uniform(-2,2,6)]:

    fk_pb = fk_pybullet(q)
    fk_my = fk_ours(q)
    print(f"\nq = {np.round(q,3)}")
    for name in my_names:
        p_pos, p_R = fk_pb[name]
        m_pos, m_R = fk_my[name]

        # 旋转误差（角度）
        ang = np.rad2deg(
              np.arccos(np.clip((np.trace(p_R.T @ m_R) - 1)/2, -1, 1)))
        # 平移误差（mm）
        dpos = np.linalg.norm(p_pos - m_pos)*1000
        print(f"{name:<9s}  Δrot={ang:6.2f}°   Δpos={dpos:6.1f} mm")

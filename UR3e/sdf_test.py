import os, time, math, random, numpy as np, torch, pybullet as p, pybullet_data
from ur_sdf.cdf.data_generator import DataGenerator
from UR3e.UR3eSim import UR3eSim

# 参数
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TEST_PTS    = 100           # 每个 q 测试多少点
N_Q_TEST      = 5             # 要测试多少个不同的 q
DIST_THRESH   = 1.0           
POINT_RADIUS  = 0.004       
ERROR_THRESH  = 0.02          

# 工具
def add_vis_point(pt, rgba=(1,0,0,1), r=POINT_RADIUS):
    vid = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba)
    cid = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cid,
                             baseVisualShapeIndex=vid, basePosition=pt)

def bullet_distance(robot_id, point_body, thresh=DIST_THRESH):
    cps = p.getClosestPoints(robot_id, point_body, distance=thresh)
    return cps[0][8] if cps else None

def check_pose(gen, sim, q_now, n_pts=50):

    q_torch = torch.tensor([q_now], dtype=torch.float32, device=gen.device)

    ws_min = torch.tensor(gen.workspace[0], device=gen.device)
    ws_max = torch.tensor(gen.workspace[1], device=gen.device)
    x_world = ws_min + torch.rand(n_pts, 3, device=gen.device) * (ws_max - ws_min)

    results = []
    for i, x in enumerate(x_world):
        x_np = x.cpu().numpy()
        d_sdf = gen.compute_sdf(x.unsqueeze(0), q_torch)[0].item()
        sphere_id = add_vis_point(x_np)
        d_blt = bullet_distance(sim.ur3e, sphere_id)
        p.removeBody(sphere_id)

        if d_blt is None:
            continue
        diff = d_sdf - d_blt
        results.append((x_np, d_sdf, d_blt, diff))

    # 误差统计
    diffs = np.array([d for (_, _, _, d) in results])
    abs_diffs = np.abs(diffs)
    print("\n========================================")
    print(f"q = {np.round(q_now, 3).tolist()}")
    for i, (pt, sdf_val, blt_val, diff) in enumerate(results):
        print(f"[{i:03d}] x={np.round(pt,3)}  sdf={sdf_val:.6f}  "
              f"bullet={blt_val:.6f}  diff={diff:.6e}")
    print("--------------------------------------------------------------")
    print(f"样本数: {len(results)}")
    print(f"平均误差: {abs_diffs.mean():.6f} m")
    print(f"最大误差: {abs_diffs.max():.6f} m")
    print(f"超过 {ERROR_THRESH*100:.0f} mm 的比例: {(abs_diffs > ERROR_THRESH).mean()*100:.1f}%")
    print("==============================================================")

def main():
    gen = DataGenerator(torch.device(DEVICE))
    sim = UR3eSim()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())


    q_min = gen.q_min.cpu().numpy()
    q_max = gen.q_max.cpu().numpy()
    for q_idx in range(N_Q_TEST):
        q_rand = np.random.uniform(q_min, q_max)  
        sim.set_joint_angles(q_rand)
        check_pose(gen, sim, q_rand, n_pts=N_TEST_PTS)


    while True:
        p.stepSimulation()
        time.sleep(1/240.)

if __name__ == "__main__":
    main()

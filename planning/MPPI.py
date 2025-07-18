import torch
import numpy as np
import pybullet

class MPPIControllet:
    def __init__(self, sim, horizon=15, num_samples=100, noise_sigma=0.1, lam=1.0):
        self.sim = sim
        self.horizon = horizon
        self.num_samples = num_samples
        self.noise_sigma = noise_sigma
        self.lam = lam
        self.control_dim = len(sim.control_joints)
        self.u_nominal = np.zeros((horizon, self.control_dim))
    
    def compute_cost(self, trajectory, target_pos):
        cost = 0.0
        # Penalize distance to target position
        cost += np.linalg.norm(trajectory[-1] - target_pos) ** 2
        save_state = trajectory[0]
        # Penalize collisions
        for state in trajectory:
            self.sim.set_joint_angles(state)
            pybullet.stepSimulation()
            if self.sim.check_collisions():
                cost += 1
        # Reset to the initial state
        self.sim.set_joint_angles(save_state)
        pybullet.stepSimulation()
        return cost
    
    def rollout(self, u_seq):
        cur_joint_angles = self.sim.get_joint_angles()
        trajectory = np.zeros((self.num_samples, self.horizon + 1, self.control_dim))
        # Generate noisy control sequences: shape (num_samples, horizon, control_dim)
        noise = np.random.normal(0, self.noise_sigma, size=(self.num_samples, self.horizon, self.control_dim))
        # Add noise to the nominal control sequence
        u_next_seq = u_seq[None, :, :] + noise
        for idx in range(self.num_samples):
            trajectory[idx, 0] = cur_joint_angles
            for t in range(1, self.horizon):
                trajectory[idx, t] = trajectory[idx, t - 1] + u_next_seq[idx, t - 1]
        return trajectory, u_next_seq
    
    def act(self, cur_pos, target_pos):

        while np.linalg.norm(cur_pos - target_pos) > 0.01:
            # Generate control sequences
            u_seq = self.u_nominal.copy()
            # Rollout trajectories
            trajectories, u_next_seq = self.rollout(u_seq)
            # Compute costs for each trajectory
            costs = np.zeros(self.num_samples)
            for idx in range(self.num_samples):
                costs[idx] = self.compute_cost(trajectories[idx], target_pos)
            
            weights = np.zeros(self.num_samples)
            all_exps_costs = np.sum(np.exp(-self.lam * costs))
            for idx in range(self.num_samples):
                weights[idx] = np.exp(-self.lam * costs[idx]) / all_exps_costs
            
            # Compute the next control
            u_next = np.sum(weights[:, None] * u_next_seq, axis=0)
            next_pos = cur_pos + u_next[0]
            self.sim.set_joint_angles(next_pos)
            cur_pos = self.sim.get_joint_angles()
            print(f"Current Position: {cur_pos}, Target Position: {target_pos}")
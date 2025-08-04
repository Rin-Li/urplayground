import pybullet
import pybullet_data
import os
import random
import numpy as np

class ObstacleScene():
    def __init__(self, scene_type="simple"):
        self.scene_type = scene_type
        self.obstacles = []

        if self.scene_type == "simple":
            size = 0.005
            collision_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[size]*3)
            visual_shape = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[size]*3, rgbaColor=[1,0,0,1])
            cube_body = pybullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                                 baseVisualShapeIndex=visual_shape, basePosition=[-0.2181723415851593, 0.18266341090202332, -0.1703624129295349])
            self.obstacles = [cube_body]

        elif self.scene_type == "complex":
            for _ in range(5):
                x = random.uniform(-0.4, 0.4)
                y = random.uniform(-0.4, 0.4)
                z = random.uniform(0.1, 0.3)
                color = [random.random(), random.random(), random.random(), 1.0]
                size = random.uniform(0.03, 0.07)
                collision_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[size]*3)
                visual_shape = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[size]*3, rgbaColor=color)
                cube_body = pybullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, 
                                                     baseVisualShapeIndex=visual_shape, basePosition=[x, y, z])
                self.obstacles.append(cube_body)

        elif self.scene_type == "ring_and_wall":
            self._create_wall_obstacle()
            self._create_ring_obstacle()

    def _create_wall_obstacle(self):
        size = 0.5  # wall is 0.5 x 0.5 meters
        step = 0.05
        z = 0.0
        wall_center = np.array([0.5, 0.0, 0.2])

        # generate grid in X-Y, then rotate and translate
        xs = np.arange(-size / 2, size / 2, step)
        ys = np.arange(-size / 2, size / 2, step)
        for x in xs:
            for y in ys:
                local = np.array([x, y, z])
                world = wall_center + self._apply_rotation(local, rot_type='wall')
                self._create_sphere_obstacle(world, radius=0.05, color=[0.8, 0.3, 0.3, 1])

    def _create_ring_obstacle(self):
        radius = 0.4
        center = np.array([0.3, 0.0, 0.45])
        theta_list = np.arange(0, 2 * np.pi, 0.2)
        for theta in theta_list:
            local = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
            world = center + self._apply_rotation(local, rot_type='ring')
            self._create_sphere_obstacle(world, radius=0.05, color=[0.3, 0.3, 0.8, 1])

    def _apply_rotation(self, vec, rot_type):
        if rot_type == 'wall':
            rot = np.array([
                [1.0, 0.0,  0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0,  0.0]
            ])
        elif rot_type == 'ring':
            rot = np.array([
                [ 0.0, 0.0, -1.0],
                [ 0.0, 1.0,  0.0],
                [ 1.0, 0.0,  0.0]
            ])
        return rot @ vec

    def _create_sphere_obstacle(self, position, radius=0.05, color=[1, 0, 0, 1]):
        col_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=radius)
        vis_shape = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=radius, rgbaColor=color)
        sphere = pybullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape,
                                          baseVisualShapeIndex=vis_shape, basePosition=position.tolist())
        self.obstacles.append(sphere)

    def create_obstacle_scene(self):
        print("Creating obstacle scene with type:", self.scene_type)
        return self.obstacles

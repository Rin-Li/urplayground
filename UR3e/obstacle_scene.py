import pybullet
import pybullet_data
import os
import random

class ObstacleScene():
    def __init__(self, scene_type="simple"):
        self.scene_type = scene_type
        self.obstacles = []
        # Simple initialization of obstacles list
        if self.scene_type == "simple":
            self.cube1 = pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube.urdf"), 
                                            [0.3, 0.3, 0.1], globalScaling=0.5)
            self.cube2 = pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube.urdf"), 
                                            [-0.3, 0.3, 0.1], globalScaling=0.5)
            self.obstacles = [self.cube1, self.cube2]

        # Complex initialization of obstacles list
        if self.scene_type == "complex":
            for _ in range(5):
                x = random.uniform(-0.4, 0.4)
                y = random.uniform(-0.4, 0.4)
                z = random.uniform(0.1, 0.3)
                color = [random.random(), random.random(), random.random(), 1.0]
                size = random.uniform(0.03, 0.07)
                collision_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[size, size, size])
                visual_shape = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=color)
                cube_body = pybullet.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision_shape, 
                                                    baseVisualShapeIndex=visual_shape, basePosition=[x, y, z])
                self.obstacles.append(cube_body)
    def create_obstacle_scene(self):
        print("Creating obstacle scene with type:", self.scene_type)
        return self.obstacles
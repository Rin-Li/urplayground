import numpy as np


data = np.load("/home/kklab-ur-robot/urplayground/UR3e/data_partial_8000.npy", allow_pickle=True).item()

print(data[4502]['x'])
print(data[4502]['q'])
print(data[4502]['idx'])
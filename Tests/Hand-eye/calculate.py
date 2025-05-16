import numpy as np
from scipy.spatial.transform import Rotation as R

R_cam2gripper = np.load("./R_cam2gripper.npy")
t_cam2gripper = np.load("./t_cam2gripper.npy")

# R_gripper2cam = R_cam2gripper.T
# t_gripper2cam = -R_gripper2cam @ t_cam2gripper

print("Cam2Gripper")
rot = R.from_matrix(R_cam2gripper)
rz, ry, rx = rot.as_euler('ZYX', degrees=True)
print(f"rx: {rx:.3f}, ry: {ry:.3f}, rz: {rz:.3f}")
print(t_cam2gripper)

camera_tool = np.array([
    t_cam2gripper[0][0],  # x
    t_cam2gripper[1][0],  # y
    t_cam2gripper[2][0],  # z
    rx,                  # rotation x
    ry,                  # rotation y
    rz                   # rotation z
])

print(camera_tool)
np.save("./camera_tool.npy", camera_tool)
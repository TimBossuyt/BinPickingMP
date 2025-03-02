import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from luxonis_camera.calibrate import CameraCalibrator
import cv2

def transform_point(point, trans_mat):
    # Convert the 3D point to homogeneous coordinates (add a 1 for the w-coordinate)
    point_homogeneous = np.array([point[0], point[1], point[2], 1])

    # Perform matrix multiplication: transformed_point = trans_mat * point_homogeneous
    transformed_point_homogeneous = np.dot(trans_mat, point_homogeneous)

    # Return the transformed point by converting back to 3D coordinates
    return transformed_point_homogeneous[:3]  # We discard the w-coordinate

## Camera parameters (for 4K resolution)
arrIntrinsics4K = np.asarray([[3082.5126953125, 0.0, 1898.798583984375],
                            [0.0, 3080.830078125, 1104.401123046875],
                            [0.0, 0.0, 1.0]])


## Adjust intrinsics to new size: 1920/1080 --> first and second row / 2
arrIntrinsics = arrIntrinsics4K.copy()
arrIntrinsics[:2, :] /= 2


oCalibrator = CameraCalibrator(arrIntrinsics)

charuco_3D_world = {
    5: [0, 0, 0],
    0: [187.5, 0, 0],
    18: [187.5, 112.5, 0],
    23: [0, 112.5, 0]
}

calib_image = cv2.imread("2025-03-01_16-15-08.jpg")

trans_mat = oCalibrator.runCalibration(calib_image, charuco_3D_world)

## Get used arrays
cam_points = oCalibrator.arrCamPoints
world_points = oCalibrator.arrWorldPoints

# Apply transformation to camera points
transformed_cam_points = np.array([transform_point(p, trans_mat) for p in cam_points])

## Visualize in 3D Space
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colors for each pair of points
colors = ['r', 'g', 'b', 'c', 'm']

# Plot points
for i in range(len(cam_points)):
    ax.scatter(*world_points[i], color=colors[i], label=f'World {i}' if i == 0 else "")
    ax.scatter(*cam_points[i], color=colors[i], marker='^', label=f'Camera {i}' if i == 0 else "")
    ax.scatter(*transformed_cam_points[i], color=colors[i], marker='s', label=f'Transformed {i}' if i == 0 else "")
    ax.plot([cam_points[i, 0], transformed_cam_points[i, 0]],
            [cam_points[i, 1], transformed_cam_points[i, 1]],
            [cam_points[i, 2], transformed_cam_points[i, 2]],
            color=colors[i], linestyle='dotted')

# Create polygons to form opaque surfaces
faces = [
    [world_points[0], world_points[1], world_points[2], world_points[3]],  # Base
    [cam_points[0], cam_points[1], cam_points[2], cam_points[3]],  # Top
]

for face in faces:
    poly = art3d.Poly3DCollection([face], alpha=0.5, color='gray')
    ax.add_collection3d(poly)

# Labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of World and Camera Points")
ax.legend()

# Show the plot
plt.show()


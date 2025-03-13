import copy

from luxonis_camera import CameraCalibrator
import cv2
import numpy as np
import open3d as o3d

####### --------------------- ####################
"""
Camera probably uses left handed coordinate frame while I try to transform it to a right handed world frame
Solution: OR using left-handed world frame or trying to convert camera's spatial coordinates to a right-handed world frame
"""
####### --------------------- ####################

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transforms a set of 3D points using a homogeneous transformation matrix.

    :param points: (N, 3) NumPy array of 3D points.
    :param transform: (4, 4) NumPy array representing the homogeneous transformation matrix.
    :return: (N, 3) NumPy array of transformed 3D points.
    """
    if points.shape[1] != 3:
        raise ValueError("Input points should have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("Transformation matrix should have shape (4, 4)")

    # Convert points to homogeneous coordinates (N, 4)
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])

    # Apply transformation
    transformed_homogeneous = (transform @ homogeneous_points.T).T

    # Convert back to 3D coordinates
    transformed_points = transformed_homogeneous[:, :3]

    return transformed_points

arrIntrinsics = np.asarray([[1541.25634765625, 0.0, 949.3992919921875], [0.0, 1540.4150390625, 552.2005615234375], [0.0, 0.0, 1.0]])

## 1. Read the image and show
img = cv2.imread("./test_input/2025-03-02_16-03-35.jpg")

cv2.imshow('Calibration image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## 2. Create calibration points
# charuco_3D_world = {0: [0, 0, 0],
#                      5: [0, 187.5, 0],
#                      23: [112.5, 187.5, 0],
#                      18: [112.5, 0, 0] }

charuco_3D_world = {0: [0, 0, 0],
                     6: [37.5, 0, 0],
                     1: [0, 37.5, 0],
                     7: [37.5, 37.5, 0],
                    2: [0, 75, 0],
                    8: [37.5, 75, 0],
                    12: [75, 0, 0],
                    }

## 3. Apply the calibrator object
oCalibrator = CameraCalibrator(arrIntrinsics)
trans_mat = oCalibrator.runCalibration(img, charuco_3D_world)

print(trans_mat)


## VERIFICATION
## Get Z-values in pointcloud coordinates vs. estimated calibration coordinates


## 1. Read pointcloud from file
pcd = o3d.io.read_point_cloud("./test_input/2025-03-02_16-03-35.ply")

# ## Left-handed --> right handed by flipping x-axis
# Convert to numpy array
points = np.asarray(pcd.points)

# Flip the x-axis
# points[:, 0] = -points[:, 0]

## Left handed luxonis --> right handed opencv (see opencv pinhole camera model)
points[:, 1] = -points[:, 1]


# Update the point cloud with the flipped coordinates
pcd.points = o3d.utility.Vector3dVector(points)


## 2. Transform and visualize
pcd_transformed = copy.deepcopy(pcd)
pcd_transformed.transform(trans_mat)


origin_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=(0, 0, 0))
origin_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=(0, 0, 0))
origin_world.transform(trans_mat)


o3d.visualization.draw_geometries([origin_camera, pcd_transformed ])
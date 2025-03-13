import copy
from luxonis_camera import CameraCalibrator
import cv2
import numpy as np
import open3d as o3d


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

## COBOT POINTS
charuco_3D_world = {0: [487, 624, 0.1],
                     18: [406, 548, -0.5],
                     23: [537, 411, -0.175],
                    5: [619, 489, 1.705],
                    9: [536, 519, -0.6],
                    13: [460, 547, -0.144],
                    100: [664, 360, 87]
                    }

## 1. Read the image and pointcloud
img = cv2.imread("./test_input/2025-03-13_15-15-44.jpg")

pcd = o3d.io.read_point_cloud("./test_input/2025-03-13_15-15-44.ply")
points = np.asarray(pcd.points)


oCalibrator = CameraCalibrator(arrIntrinsics)
trans_mat, scale = oCalibrator.runCalibration(img, pcd, charuco_3D_world)

## 2. Visualize
pcd_transformed = copy.deepcopy(pcd)
pcd_transformed.transform(trans_mat)

origin_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=(0, 0, 0))

o3d.visualization.draw_geometries([origin_camera, pcd_transformed ])

## Cobot test
for id, coords in oCalibrator.dictCamera3DPoints.items():
    print("==========================")
    print(f"Transformation for corner: {id}")
    print("Coordinates in camera space")
    print(coords)
    print("Coordinates in cobot space")
    point_transformed = transform_points(coords.reshape(1, 3)*scale, trans_mat)
    print(point_transformed)
    print("==========================")

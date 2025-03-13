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
img_reproject = copy.deepcopy(img)

pcd = o3d.io.read_point_cloud("./test_input/2025-03-13_15-15-44.ply")

oCalibrator = CameraCalibrator(arrIntrinsics)
trans_mat = oCalibrator.runCalibration(img, pcd, charuco_3D_world)

## 2. Visualize
pcd_transformed = copy.deepcopy(pcd)
pcd_transformed.transform(trans_mat)

origin_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=(0, 0, 0))

o3d.visualization.draw_geometries([origin_camera, pcd_transformed ])

## Cobot test

# Get all XYZ coordinates from pointcloud from index range
kernel_size = 6

points = np.asarray(pcd.points)
points[:, 1] = -points[:, 1]
pcd_points = points.reshape(1080, 1920, 3)

for id, coords in oCalibrator.dictCameraImagePoints.items():
    print("==========================")
    print(f"Transformation for corner: {id}")
    x, y = int(coords[0]), int(coords[1])
    ## Get the 3D coordinates on that point
    # Get window with given kernel size
    # = all surrounding points (Don't hit the boundaries of the image!)
    x_min = int(x - kernel_size/2)
    x_max = int(x + kernel_size/2)
    y_min = int(y - kernel_size/2)
    y_max = int(y + kernel_size/2)

    ## Returns shape 6, 6, 3
    point_window = pcd_points[y_min:y_max, x_min:x_max]
    # print(point_window)

    point_3d_mean = np.mean(point_window.reshape(-1, 3), axis=0)
    # print("Mean")
    # print(point_3d_mean)
    point_3d = point_3d_mean
    print("Coordinates in camera space")
    print(point_3d)
    print("Coordinates in cobot space")
    point_transformed = transform_points(point_3d.reshape(1, 3), trans_mat)
    print(point_transformed)
    print("==========================")

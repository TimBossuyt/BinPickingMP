import open3d as o3d
import os
import numpy as np

from pose_estimation import Scene
from pose_estimation import SettingsManager
from pose_estimation.utils import display_point_clouds


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


sPath = "../settings.json"

assert os.path.exists(sPath), "Could not find settings file"

oSm = SettingsManager(sPath)
pcd = o3d.io.read_point_cloud("./test_input/2025-04-03_11-48-09.ply")


## Just for debugging first transform all the points (as camera was not calibrated)
twc = np.load("../../CalibrationData/tcw.npy")
twc_scale = float(np.load("../../CalibrationData/scale_tcw.npy"))

arrPoints = np.asarray(pcd.points)
## Flip to right handed
arrPoints[:, 1] = -arrPoints[:, 1]
arrPoints = transform_points(arrPoints*twc_scale, twc)

pcd_transformed = o3d.geometry.PointCloud()
pcd_transformed.points = o3d.utility.Vector3dVector(arrPoints)
pcd_transformed.colors = o3d.utility.Vector3dVector(pcd.colors)

display_point_clouds([pcd_transformed], "Pointcloud - transformed", False, True, 100)

scene = Scene(pcd_transformed, oSm)
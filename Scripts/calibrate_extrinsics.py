import cv2
import open3d as o3d
import numpy as np

from pose_estimation import ExtrinsicTransformationAruco


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

pcdScene = o3d.io.read_point_cloud(
    "../PointCloudImages/PointClouds_2024-12-07_22-48-37/2024-12-07_22-48-53/PointCloud_2024-12-07_22-48-53.ply")

oTransformation = ExtrinsicTransformationAruco(pcdScene, 1080, 1920, aruco_dict)

## Save transformation matrix
np.save("../CalibrationData/arrTransMat.npy", oTransformation.arrTransMat)
print("Transformation matrix saved to 'arrTransMat.npy'")
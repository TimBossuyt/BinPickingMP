import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os


def extract_translation_rotation(transform_matrix):
    # Extract the translation part (x, y, z) from the 4th column of the matrix
    translation = transform_matrix[:3, 3]
    x, y, z = translation[0], translation[1], translation[2]

    # Extract the rotation matrix (upper-left 3x3 sub-matrix)
    rotation_matrix = transform_matrix[:3, :3]

    # Use scipy to get Euler angles from the rotation matrix (in degrees)
    rotation = R.from_matrix(rotation_matrix)
    anglex, angley, anglez = rotation.as_euler('xyz', degrees=True)

    # Print the results nicely to the terminal
    print("Translation (x, y, z):")
    print(f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}")

    print("\nRotation angles (degrees):")
    print(f"Angle around x-axis: {anglex:.2f}°")
    print(f"Angle around y-axis: {angley:.2f}°")
    print(f"Angle around z-axis: {anglez:.2f}°")

    return x, y, z, anglex, angley, anglez


def display_point_clouds(point_clouds, window_title, show_normals=False):
    """
    Displays a list of Open3D point clouds in a single visualization window.

    Parameters:
    - point_clouds: List of Open3D point cloud objects
    """
    # Check if the list is not empty
    if not point_clouds:
        print("Error: The list of point clouds is empty.")
        return

    # Visualize all point clouds together
    o3d.visualization.draw_geometries(point_clouds,
                                      window_name=window_title,
                                      width=800,
                                      height=600,
                                      point_show_normal=show_normals)

def extract_square_region(array, center, side_length):
    """
    Extracts a square region from a 2D or 3D NumPy array based on a center coordinate and side length.

    Parameters:
    array (np.ndarray): The input array, e.g., an image of shape (H, W, C).
    center (tuple): The (x, y) coordinates of the center of the square.
    side_length (int): The length of the sides of the square.

    Returns:
    np.ndarray: The square region extracted from the input array.
    """
    center_x, center_y = center
    half_side = side_length // 2

    # Calculate the bounds of the square, ensuring they're within the array's bounds
    top_left_x = max(center_x - half_side, 0)
    top_left_y = max(center_y - half_side, 0)
    bottom_right_x = min(center_x + half_side, array.shape[1])
    bottom_right_y = min(center_y + half_side, array.shape[0])

    # Extract the square region
    square_region = array[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return square_region


class PoseEstimator:
    def __init__(self, iRelativeSamplingStepModel, iRelativeDistanceStepModel, iNumOfAngles):
        self.oDetector = cv2.ppf_match_3d_PPF3DDetector(iRelativeSamplingStepModel, iRelativeDistanceStepModel, iNumOfAngles)
        self.xModelLoaded = False
        self.xModelTrained = False
        self.xSceneLoaded = False
        self.oBestResult = None
        self.oFileStorage = None

        self.oIcp = cv2.ppf_match_3d_ICP(500)

    def loadModel(self, sFilename, iContainsNormals):
        self.oModel = cv2.ppf_match_3d.loadPLYSimple(sFilename, iContainsNormals)
        ## voorwaarde toevoegen
        if self.oModel is not None:
            self.xmodelLoaded = True
            print("Model loaded")

    def loadScene(self, sFilename, iContainsNormals):
        self.oScene = cv2.ppf_match_3d.loadPLYSimple(sFilename, iContainsNormals)
        if self.oScene is not None:
            self.xSceneLoaded = True
            print("Scene Loaded")

    def trainModel(self):
        print("Training model...")
        self.oDetector.trainModel(self.oModel)
        print("Training done")
        self.xModelTrained = True

    def match(self, iRelativeSceneSampleStep, iRelativeSceneDistance):
        print("Matching...")
        oResultsTemp = self.oDetector.match(self.oScene, iRelativeSceneSampleStep, iRelativeSceneDistance)

        print("Applying ICP")
        _, self.oResults = self.oIcp.registerModelToScene(self.oModel, self.oScene, oResultsTemp)

    def getBestResult(self):
        ## Not the best method to evaluate best estimation
        for i, oResult in enumerate(self.oResults[:5]):
            if self.oBestResult is None or oResult.residual < self.oBestResult.residual:
                self.oBestResult = oResult

        return self.oBestResult.pose

    def getNPoses(self, iN):
        return self.oResults[:iN]


    def showResults(self, iNumberOfResults):
        for i, oResult in enumerate(self.oResults[:iNumberOfResults]):
            oResult.printPose()

            oModelTransformed = cv2.ppf_match_3d.transformPCPose(self.oModel, oResult.pose)

            ## Convert Matlike (OpenCV) point clouds to Open3D pointclouds
            oSceneO3D = o3d.geometry.PointCloud()
            oSceneO3D.points = o3d.utility.Vector3dVector(self.oScene[:, :3])
            oSceneO3D.paint_uniform_color((0, 1, 0))

            oModelTransformedO3D = o3d.geometry.PointCloud()
            oModelTransformedO3D.points = o3d.utility.Vector3dVector(oModelTransformed[:, :3])
            oModelTransformedO3D.paint_uniform_color((1, 0, 0))

            oOrigin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
            oOrigin_transform = oOrigin.transform(oResult.pose)

            ## Visualize
            o3d.visualization.draw_geometries([oSceneO3D, oModelTransformedO3D, oOrigin, oOrigin_transform],
                                              f"Result no. {i}", width=800, height=600)


from luxonis_camera import Camera
import logging.config
import datetime
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d

TEST_IMG = "./2025-05-15_10-46-56.jpg"
TEST_PCD = "./2025-05-15_10-46-56.ply"



def load_organized_pointcloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points).reshape((1080, 1920, 3))

    points[:, :, 1] *= -1
    # points[:, :, 1] *= -1
    # points[:, :, 2] *= 50
    return points

def transform_points(R, t, points):
    """Transform 3D points from camera to world space using R, t"""
    return (R.T @ (points.T - t)).T


np.set_printoptions(precision=3, suppress=True)

########## Logging setup ##########
## Generate ISO 8601 timestamped filename
log_filename = datetime.datetime.now().strftime("log_%Y-%m-%dT%H-%M-%S.log")

logger = logging.getLogger("Main")
###################################

########## Camera setup ##########
oCamera = Camera(5)
##################################

sMxId = "1844301011B546F500"

## Start camera and wait for the camera to connect
oCamera.Connect(sMxId)

while not oCamera.bConnected:
    continue

# Save the camera intrinsics
print("=== Calibration from camera ===")
print("Camera Matrix (Intrinsics):\n", oCamera.arrCameraMatrix)
print("Distortion Coefficients:\n", oCamera.distortion)

oCamera.Disconnect()

## Try to calibrate the camera using the charuco board
oArucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
charuco_board = cv2.aruco.CharucoBoard(
    size=(7, 5),
    markerLength=25,
    squareLength=37.5,
    dictionary=oArucoDict
)

detector = cv2.aruco.CharucoDetector(charuco_board)

all_corners = []
all_ids = []
image_size = (1920, 1080)

object_points = charuco_board.getChessboardCorners()

image = cv2.imread(TEST_IMG)
charuco_corners, charuco_ids, _, _  = detector.detectBoard(image=image)

pc_xyz = load_organized_pointcloud(TEST_PCD)

# Get 3D positions of each detected Charuco corner
charuco_points_3D = []
valid_expected_points = []

for i, corner in enumerate(charuco_corners):
    x, y = corner.ravel().astype(int)
    point = pc_xyz[y, x]
    if np.all(np.isfinite(point)) and np.linalg.norm(point) > 0:
        charuco_points_3D.append(point)
        valid_expected_points.append(object_points[charuco_ids[i][0]])  # 3D board point

charuco_points_3D = np.array(charuco_points_3D)
valid_expected_points = np.array(valid_expected_points)

_, rvec, tvec = cv2.solvePnP(object_points, charuco_corners, oCamera.arrCameraMatrix, None)

# Compare results
print("\n=== Extrinsics with Luxonis Calibration Parameters ===")
print("Rotation Vector:\n", rvec)
print("Translation Vector:\n", tvec)

R, _ = cv2.Rodrigues(rvec)
t = tvec.reshape(3, 1)

trans_mat = np.eye(4)
trans_mat[:3, :3] = R
trans_mat[:3, 3] = t.flatten()
trans_mat = np.linalg.inv(trans_mat)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_xyz.reshape(-1, 3))
pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
pcd.transform(trans_mat)

# Define min and max bounds (adjust as needed)
min_bound = np.array([-100, -100, -500])
max_bound = np.array([500, 500, 500])
# Create bounding box
aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

# Crop the point cloud using the bounding box
pcd_filtered = pcd.crop(aabb)


# Create coordinate frame (size = 0.1 can be adjusted)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

# Visualize
o3d.visualization.draw_geometries([pcd_filtered, coord_frame])


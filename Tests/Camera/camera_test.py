from luxonis_camera import Camera
import logging.config
import datetime
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D

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

# Get image file list
image_paths = os.listdir("./charuco_viewpoints/")

image_list = []
for image_path in image_paths:
    image_list.append(cv2.imread(os.path.join("./charuco_viewpoints", image_path)))


for image in image_list:
    charuco_corners, charuco_ids, _, _  = detector.detectBoard(image=image)

    # print(f"Found {len(charuco_corners)} corners for image")

    all_corners.append(charuco_corners)
    all_ids.append(charuco_ids)


flags = (
    cv2.CALIB_RATIONAL_MODEL |
    cv2.CALIB_THIN_PRISM_MODEL |
    cv2.CALIB_TILTED_MODEL
)

## Calibrate the camera
retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags
)

if retval:
        print("\n=== Calibration Results ===")
        print("Camera Matrix (Intrinsics):\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs[0])
else:
    print("Calibration failed.")

oCamera.Disconnect()

## Check undistortion of the images
K_factory = np.asarray(oCamera.arrCameraMatrix)
D_factory = np.zeros_like(oCamera.distortion)

K_charuco = np.asarray(camera_matrix)
# D_charuco = np.asarray(dist_coeffs)
D_charuco = np.zeros_like(D_factory)

object_points = charuco_board.getChessboardCorners()
# print(object_points)

# Run solvePnP for both on fixed camera position
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


success1, rvec1, tvec1 = cv2.solvePnP(object_points, charuco_corners, K_factory, D_factory)
success2, rvec2, tvec2 = cv2.solvePnP(object_points, charuco_corners, K_charuco, D_charuco)


# Compare results
print("\n=== Extrinsics with Factory Calibration ===")
print("Rotation Vector:\n", rvec1)
print("Translation Vector:\n", tvec1)

print("\n=== Extrinsics with Charuco Calibration ===")
print("Rotation Vector:\n", rvec2)
print("Translation Vector:\n", tvec2)

# Ensure both tvecs are (3,) shaped arrays
t1_flat = tvec1.flatten()
t2_flat = tvec2.flatten()

# Compute component-wise differences
diff_tvec = t2_flat - t1_flat

print("\n=== Translation Differences (Charuco - Factory) ===")
print(f"ΔX: {diff_tvec[0]:.2f} mm")
print(f"ΔY: {diff_tvec[1]:.2f} mm")
print(f"ΔZ: {diff_tvec[2]:.2f} mm")

# Flatten rotation vectors
r1 = rvec1.flatten()
r2 = rvec2.flatten()

# Component-wise difference
diff_rvec = r2 - r1

print("\n=== Rotation Vector Differences (Charuco - Factory) ===")
print(f"Δrx: {diff_rvec[0]:.4f} rad")
print(f"Δry: {diff_rvec[1]:.4f} rad")
print(f"Δrz: {diff_rvec[2]:.4f} rad")


def draw_camera(ax, R, t, label, color='r'):
    """
    Draw a simple camera coordinate frame at position t with rotation R.
    """
    # Camera axes in camera space
    axis_length = 50
    cam_axes = np.float32([[axis_length, 0, 0],    # X
                           [0, axis_length, 0],    # Y
                           [0, 0, axis_length]])   # Z

    # Transform to world coordinates
    cam_axes_world = R @ cam_axes.T + t
    origin = t.flatten()

    # Plot the camera coordinate axes
    ax.plot([origin[0], cam_axes_world[0, 0]], [origin[1], cam_axes_world[1, 0]], [origin[2], cam_axes_world[2, 0]], color='r')  # X axis
    ax.plot([origin[0], cam_axes_world[0, 1]], [origin[1], cam_axes_world[1, 1]], [origin[2], cam_axes_world[2, 1]], color='g')  # Y axis
    ax.plot([origin[0], cam_axes_world[0, 2]], [origin[1], cam_axes_world[1, 2]], [origin[2], cam_axes_world[2, 2]], color='b')  # Z axis
    ax.text(origin[0], origin[1], origin[2], label, fontsize=10)


# Assuming you have already run solvePnP to get rvec1, tvec1 (factory) and rvec2, tvec2 (charuco)
R1, _ = cv2.Rodrigues(rvec1)  # Factory
R2, _ = cv2.Rodrigues(rvec2)  # Charuco

t1 = tvec1.reshape(3, 1)
t2 = tvec2.reshape(3, 1)

# 3D plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Charuco board at origin (XY plane)
board_size = 260  # width of board in meters (adjust based on your board)
square_size = 37.5
corners = np.array([
    [0, 0, 0],
    [board_size, 0, 0],
    [board_size, board_size, 0],
    [0, board_size, 0]
])
for i in range(4):
    j = (i + 1) % 4
    ax.plot([corners[i][0], corners[j][0]], [corners[i][1], corners[j][1]], [0, 0], 'k--')

# Draw cameras
draw_camera(ax, R1, t1, "Factory", color='purple')
draw_camera(ax, R2, t2, "Charuco", color='orange')

charuco_in_world_factory = transform_points(R1, t1, charuco_points_3D)
charuco_in_world_charuco = transform_points(R2, t2, charuco_points_3D)


# Axes settings
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera Pose Comparison")
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=45)
plt.show()


expected = charuco_board.getChessboardCorners()

# Error in board frame
err_factory = np.linalg.norm(charuco_in_world_factory - valid_expected_points, axis=1)
err_charuco = np.linalg.norm(charuco_in_world_charuco - valid_expected_points, axis=1)

print("\n--- Mean error in board frame ---")
print(f"Factory: {np.mean(err_factory):.2f} mm")
print(f"Charuco: {np.mean(err_charuco):.2f} mm")

for i, (e1, e2) in enumerate(zip(err_factory, err_charuco)):
    print(f"Corner {i}: Factory error = {e1:.2f} mm, Charuco error = {e2:.2f} mm")


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Expected board points (ground truth, Z=0 plane)
ax.scatter(valid_expected_points[:, 0], valid_expected_points[:, 1], valid_expected_points[:, 2],
           c='k', marker='o', label='Expected (Board Frame)')

# Factory-calculated points
ax.scatter(charuco_in_world_factory[:, 0], charuco_in_world_factory[:, 1], charuco_in_world_factory[:, 2],
           c='purple', marker='^', label='Factory Transformed')

# Charuco-calculated points
ax.scatter(charuco_in_world_charuco[:, 0], charuco_in_world_charuco[:, 1], charuco_in_world_charuco[:, 2],
           c='orange', marker='s', label='Charuco Transformed')

# Connect corresponding points (optional)
for i in range(len(valid_expected_points)):
    ax.plot(
        [valid_expected_points[i, 0], charuco_in_world_factory[i, 0]],
        [valid_expected_points[i, 1], charuco_in_world_factory[i, 1]],
        [valid_expected_points[i, 2], charuco_in_world_factory[i, 2]],
        'purple', alpha=0.3
    )
    ax.plot(
        [valid_expected_points[i, 0], charuco_in_world_charuco[i, 0]],
        [valid_expected_points[i, 1], charuco_in_world_charuco[i, 1]],
        [valid_expected_points[i, 2], charuco_in_world_charuco[i, 2]],
        'orange', alpha=0.3
    )

ax.set_title("3D Comparison of Charuco vs Factory Transformed Points")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.legend()
ax.grid(True)
ax.view_init(elev=30, azim=-60)
plt.tight_layout()
plt.show()


from fairino import Robot
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging.config
import cv2
from luxonis_camera import Camera
import datetime
import threading
import open3d as o3d
import time
import imutils

np.set_printoptions(precision=3, suppress=True)

oCobot = Robot.RPC('192.168.58.2')

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

def preview_loop(target_fps=3):
    frame_interval = 1.0 / target_fps  # Time per frame
    last_frame_time = time.time()
    frame = None
    try:
        while True:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                frame = oCamera.getCvImageFrame()  # Get the frame
                if frame is not None:
                    cv2.imshow("Camera", imutils.resize(frame, width=500))  # Show the frame
                last_frame_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()

thrPreview = threading.Thread(target=preview_loop, daemon=True)
thrPreview.start()

camera_tool = tuple(np.load("./camera_tool.npy"))

ret = oCobot.SetToolCoord(10, camera_tool, 0, 0, 0, 0)
print("Setting TCP:", ret)

def pose_to_homogeneous_matrix(x, y, z, rx, ry, rz, degrees=True, order='ZYX'):
    R_bt = R.from_euler(order, [rz, ry, rx], degrees=degrees).as_matrix()
    t_bt = np.array([[x], [y], [z]])

    T_bt = np.eye(4)
    T_bt[:3, :3] = R_bt
    T_bt[:3, 3:] = t_bt
    return T_bt


## Take pointcloud picture
def transform_and_display_colored_pointcloud_in_base(pcd, T_camera2base):
    """
    Transforms a colored point cloud from camera frame to base frame and displays it.

    Args:
        pcd (o3d.geometry.PointCloud): Organized point cloud with color
        T_camera2base (np.ndarray): 4x4 homogeneous transformation matrix
    """
    # Load and reshape organized point cloud
    points = np.asarray(pcd.points).reshape((1080, 1920, 3))
    colors = np.asarray(pcd.colors).reshape((1080, 1920, 3))[:, :, [2, 1, 0]]


    # Flip Y-axis in 3D points
    points[:, :, 1] *= -1

    # Flatten for processing
    flat_points = points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)

    # Homogeneous coordinates
    points_hom = np.hstack([flat_points, np.ones((flat_points.shape[0], 1))])

    # Transform to base frame
    transformed_points = (T_camera2base @ points_hom.T).T[:, :3]

    # Create transformed colored point cloud
    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(transformed_points)
    pcd_base.colors = o3d.utility.Vector3dVector(flat_colors)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

    # Define min and max bounds (adjust as needed)
    min_bound = np.array([-1000, -1000, -500])
    max_bound = np.array([1000, 1000, 500])
    # Create bounding box
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Crop the point cloud using the bounding box
    pcd_filtered = pcd_base.crop(aabb)


    # Display
    o3d.visualization.draw_geometries([pcd_filtered, coord_frame], window_name="Colored PointCloud in Base Frame")

    return pcd_filtered

all_pcds = []

for i in range(5):
    input("Press enter to continue")
    pcd = oCamera.getColoredPointCloud()

    _, tool_base = oCobot.GetActualTCPPose()
    print("Getting TCP:", _)
    print("Tool Base:", tool_base)
    x, y, z, rx, ry, rz = tool_base

    T_camera2base = pose_to_homogeneous_matrix(x, y, z, rx, ry, rz)
    all_pcds.append(transform_and_display_colored_pointcloud_in_base(pcd, T_camera2base))

# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
o3d.visualization.draw_geometries(all_pcds, window_name="Assembled PointCloud in Base Frame")








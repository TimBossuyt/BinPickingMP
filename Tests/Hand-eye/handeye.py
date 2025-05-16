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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def preview_loop(target_fps=5):
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

## 1. Set the default TCP
print(oCobot.GetActualTCPNum())

error = oCobot.SetSpeed(50) # Set the global speed. Manual mode and automatic mode are set independently
print("Set global speed:",error)

ret = oCobot.SetToolCoord(0, (0, 0, 0), 0, 0, 0, 0)
print("Setting TCP:", ret)

# ret = oCobot.GetActualTCPPose()
# print("Current tool pose relative to base:", ret)

## 1. Move to basic position
# Target Cartesian pose: [x, y, z, rx, ry, rz]
pose = (350, 660, 567, 180, 0.26, -90)

# Get the joint angles using inverse kinematics
_, joint_pos = oCobot.GetInverseKin(0, pose)
print("Joint positions:", joint_pos)

# Move the robot using joint motion to the calculated joint positions
ret = oCobot.MoveJ(joint_pos, 0, 0)
print("MoveJ result:", ret)


def pose_to_homogeneous_matrix(x, y, z, rx, ry, rz, degrees=True, order='ZYX'):
    R_bt = R.from_euler(order, [rz, ry, rx], degrees=degrees).as_matrix()
    t_bt = np.array([[x], [y], [z]])

    T_bt = np.eye(4)
    T_bt[:3, :3] = R_bt
    T_bt[:3, 3:] = t_bt
    return T_bt

## Move to a position where the board is visible
oCobot.SetToolCoord(0, (0, 0, 0), 0, 0, 0, 0)


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

gripper2base_transforms = []
target2cam_transforms = []

def generate_orbit_poses(center, radius, num_positions):
    poses = []
    for i in range(num_positions):
        azimuth = 2 * np.pi * i / num_positions
        elevation = np.radians(60)  # fixed angle, 30 degrees down

        # Convert spherical to Cartesian coordinates
        x = center[0] + radius * np.cos(elevation) * np.cos(azimuth)
        y = center[1] + radius * np.cos(elevation) * np.sin(azimuth)
        z = center[2] + radius * np.sin(elevation)

        position = np.array([x, y, z])

        # Direction vector from tool to target
        direction = center - position
        direction /= np.linalg.norm(direction)

        # Compute rotation matrix so z-axis points to direction
        z_axis = direction
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1, 0, 0])  # arbitrary orthogonal axis if aligned
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
        r = R.from_matrix(R_matrix).as_euler('xyz', degrees=True)

        poses.append((x, y, z, *r))
    return poses


# Define center point and radius
center_point = np.array([350, 660, 20])
radius = 500  # mm
num_positions = 10

orbit_poses = generate_orbit_poses(center_point, radius, num_positions)
orbit_poses = orbit_poses[5:] + orbit_poses[:5]

positions = np.array([pose[:3] for pose in orbit_poses])

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the target (center) point
ax.scatter(*center_point, c='red', s=100, label='Target Point')

# Plot the tool positions
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', label='Tool Positions')

# Draw dashed lines from each tool position to the target
for pos in positions:
    ax.plot([center_point[0], pos[0]],
            [center_point[1], pos[1]],
            [center_point[2], pos[2]], 'gray', linestyle='--')

# Labels and view settings
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Tool Orbit Around Target Point')
ax.legend()
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

for i in range(10):
    # print(f"Moving to pose {i+1}/{num_positions}: {pose}")
    # try:
    #     _, joint_angles = oCobot.GetInverseKin(0, pose)
    #     ret = oCobot.MoveJ(joint_angles, 0, 0)
    # except:
    #     continue
    # time.sleep(2)  # Wait for robot to settle
    input("Press enter to capture next pose")
    try:
        ## Gets tool position relative to base
        _, tool_base = oCobot.GetActualTCPPose()
        x, y, z, rx, ry, rz = tool_base

        T_gripper2base = pose_to_homogeneous_matrix(x, y, z, rx, ry, rz)

        # T_base2gripper = np.linalg.inv(T_gripper2base)

        ## Get target 2 cam transformation
        image = oCamera.getCvImageFrame()
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(image=image)

        matched_obj_pts = []
        matched_img_pts = []

        for i, id in enumerate(charuco_ids):
            idx = int(id[0])  # charuco_ids is a (N,1) array
            matched_obj_pts.append(object_points[idx])
            matched_img_pts.append(charuco_corners[i][0])  # charuco_corners[i] is [[x, y]]

        matched_obj_pts = np.array(matched_obj_pts, dtype=np.float32)
        matched_img_pts = np.array(matched_img_pts, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=matched_obj_pts,  # Nx3
            imagePoints=matched_img_pts,  # Nx2
            cameraMatrix=oCamera.arrCameraMatrix,
            distCoeffs=None,
        )

        if success:
            R_tc, _ = cv2.Rodrigues(rvec)
            T_target2cam = np.eye(4)
            T_target2cam[:3, :3] = R_tc
            T_target2cam[:3, 3:] = tvec

        else:
            print("Pose estimation failed.")
            raise Exception

        gripper2base_transforms.append(T_gripper2base)
        target2cam_transforms.append(T_target2cam)
    except:
        print("Something went wrong")

# oCamera.Disconnect()

def extract_R_t_from_T(T_list):
    R_list = []
    t_list = []
    for T in T_list:
        R = T[:3, :3]
        t = T[:3, 3].reshape(3, 1)
        R_list.append(R)
        t_list.append(t)
    return R_list, t_list

R_gripper2base, t_gripper2base = extract_R_t_from_T(gripper2base_transforms)
R_target2cam, t_target2cam = extract_R_t_from_T(target2cam_transforms)

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print(R_cam2gripper)
print(t_cam2gripper)

np.save("R_cam2gripper.npy", R_cam2gripper)
np.save("t_cam2gripper.npy", t_cam2gripper)

thrPreview.join()
oCamera.Disconnect()












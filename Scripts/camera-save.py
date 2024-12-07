import depthai as dai
from time import sleep
import numpy as np
import cv2
import time
import os
import sys
import open3d as o3d

FPS = 10
SAVE_INTERVAL = 5  # Time interval to save point cloud files (in seconds)

class FPSCounter:
    def __init__(self):
        self.frameCount = 0
        self.fps = 0
        self.startTime = time.time()

    def tick(self):
        self.frameCount += 1
        if self.frameCount % 10 == 0:
            elapsedTime = time.time() - self.startTime
            self.fps = self.frameCount / elapsedTime
            self.frameCount = 0
            self.startTime = time.time()
        return self.fps


# Function to create the main directory with the current timestamp
def create_main_directory():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.join("../PointCloudImages", f"PointClouds_{timestamp}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


# Function to create a subdirectory for each point cloud and image capture
def create_subdirectory(main_dir):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    subfolder_name = f"{timestamp}"
    subfolder_path = os.path.join(main_dir, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    return subfolder_path, timestamp


# Create the pipeline and set up the nodes
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)
sync = pipeline.create(dai.node.Sync)
xOut = pipeline.create(dai.node.XLinkOut)
xOut.input.setBlocking(False)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(FPS)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setCamera("left")
monoLeft.setFps(FPS)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setCamera("right")
monoRight.setFps(FPS)


## Stereodepth configuration
depth.setLeftRightCheck(True)
# depth.setExtendedDisparity(True)
depth.setSubpixel(True)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)


monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(pointcloud.inputDepth)
camRgb.isp.link(sync.inputs["rgb"])
pointcloud.outputPointCloud.link(sync.inputs["pcl"])
sync.out.link(xOut.input)
xOut.setStreamName("out")

# Main loop to process the data and save point clouds and images every 3 seconds
with dai.Device(pipeline) as device:
    isRunning = True


    def key_callback(vis, action, mods):
        global isRunning
        if action == 0:
            isRunning = False


    device.setIrLaserDotProjectorIntensity(1)

    q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_action_callback(81, key_callback)  # Q to quit visualization
    pcd = o3d.geometry.PointCloud()
    coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
    vis.add_geometry(coordinateFrame)

    main_folder = create_main_directory()  # Create the main folder for saving point clouds and images
    first = True
    fpsCounter = FPSCounter()
    last_save_time = time.time()  # Track the last time a point cloud and image were saved

    while isRunning:
        inMessage = q.get()
        inColor = inMessage["rgb"]
        inPointCloud = inMessage["pcl"]
        cvColorFrame = inColor.getCvFrame()

        # Convert the frame to RGB
        cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)
        fps = fpsCounter.tick()

        # Display the FPS on the frame
        cv2.imshow("color", cvColorFrame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Process point cloud and image saving
        if inPointCloud:
            t_before = time.time()
            points = inPointCloud.getPoints().astype(np.float64)
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if first:
                vis.add_geometry(pcd)
                first = False
            else:
                vis.update_geometry(pcd)

            # Save the point cloud and image every 3 seconds
            if time.time() - last_save_time > SAVE_INTERVAL:
                subfolder_path, timestamp = create_subdirectory(main_folder)  # Create a subfolder with ID and timestamp

                # Save the point cloud as .ply
                pointcloud_path = os.path.join(subfolder_path, f"PointCloud_{timestamp}.ply")
                o3d.io.write_point_cloud(pointcloud_path, pcd)
                print(f"Point cloud saved: {pointcloud_path}")

                # Save the RGB image as .png
                image_path = os.path.join(subfolder_path, f"image_{timestamp}.png")
                cv2.imwrite(image_path, cvColorFrame)
                print(f"Image saved: {image_path}")

                # Reset the save timer
                last_save_time = time.time()

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

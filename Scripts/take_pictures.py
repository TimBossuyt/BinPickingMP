from luxonis_camera import Camera, getConnectedDevices
import cv2
import threading
import time
import imutils
from datetime import datetime
import os
import open3d as o3d



oCamera = Camera(iFPS=5)
sMxId = "1844301011B546F500"
# print(getConnectedDevices())

# Create the images directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

## Start camera and wait for the camera to connect
oCamera.Connect(sMxId)

while not oCamera.bConnected:
    continue

def preview_loop(target_fps=5):
    cv2.namedWindow("Camera Preview", cv2.WINDOW_AUTOSIZE)
    frame_interval = 1.0 / target_fps  # Time per frame
    last_frame_time = time.time()
    frame = None
    try:
        while True:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                frame = oCamera.getCvImageFrame()  # Get the frame
                if frame is not None:
                    cv2.imshow("Camera Preview", imutils.resize(frame, width=500))  # Show the frame
                last_frame_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame is not None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"images/{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")

                pcd = oCamera.getColoredPointCloud()
                pointcloud_path = os.path.join("pointclouds", f"{timestamp}.ply")
                o3d.io.write_point_cloud(pointcloud_path, pcd)


    finally:
        cv2.destroyAllWindows()

thrPreview = threading.Thread(target=preview_loop, daemon=True)
thrPreview.start()

thrPreview.join()

oCamera.Disconnect()





import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


## Image resolution: 1920x1080
iWidthImage = 1920
iHeightImage = 1080

sScenePath = "PointCloudImages/PointClouds_2024-12-07_18-33-23/2024-12-07_18-33-34/PointCloud_2024-12-07_18-33-34.ply"

## Loading scene
pcdScene = o3d.io.read_point_cloud(sScenePath)

## Points as numpy array
arrPoints = np.asarray(pcdScene.points).reshape(iHeightImage, iWidthImage, 3)

## Colors per point as numpy array
arrColors = np.asarray(pcdScene.colors).reshape(iHeightImage, iWidthImage, 3) # (1920, 1080, 3), RGB
arrColors = (arrColors * 255).astype(np.uint8)
arrColorsBGR = cv2.cvtColor(arrColors, cv2.COLOR_RGB2BGR) ## BGR Format for OpenCV

# Display the image
cv2.imshow("Scene image (scaled)", cv2.resize(arrColorsBGR, (0,0), fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()

## ArUco detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(arrColorsBGR)

## Display
# Draw the detected markers on the image
image_with_markers = cv2.aruco.drawDetectedMarkers(arrColorsBGR.copy(), markerCorners, markerIds)

# Display the image with markers
cv2.imshow("Detected ArUco Markers", cv2.resize(image_with_markers, (0,0), fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
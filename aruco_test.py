import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def display_point_clouds(arrPointClouds, sWindowTitle, bShowNormals=False, bShowOrigin=False, iOriginSize=0):
    """
    Displays a list of Open3D point clouds in a single visualization window.

    Parameters:
    - point_clouds: List of Open3D point cloud objects to display.
    - window_title: Title of the visualization window.
    - show_normals: Boolean flag to indicate whether to display normals.
    - show_origin: Boolean flag to indicate whether to display the origin.
    - iOriginSize: Size of the origin coordinate axes (if show_origin is True).
    """
    # Check if the list is not empty
    if not arrPointClouds:
        print("Error: The list of point clouds is empty.")
        return

    arrGeometries = arrPointClouds.copy()

    if bShowOrigin:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=iOriginSize)
        arrGeometries.append(origin)

    # Visualize all point clouds together
    o3d.visualization.draw_geometries(arrGeometries,
                                      window_name=sWindowTitle,
                                      width=800,
                                      height=600,
                                      point_show_normal=bShowNormals)

## Image resolution: 1920x1080
iWidthImage = 1920
iHeightImage = 1080

sScenePath = "PointCloudImages/PointClouds_2024-12-07_18-33-23/2024-12-07_18-33-34/PointCloud_2024-12-07_18-33-34.ply"

## Loading scene
pcdScene = o3d.io.read_point_cloud(sScenePath)

## Voxel down
pcdSceneDown = pcdScene.voxel_down_sample(voxel_size=10)

# display_point_clouds([pcdScene], "Scene", False, True, 100)

arrPoints = np.asarray(pcdScene.points).reshape(iHeightImage, iWidthImage, 3)

arrColors = np.asarray(pcdScene.colors).reshape(iHeightImage, iWidthImage, 3) # (1920, 1080, 3), RGB
arrColors = (arrColors * 255).astype(np.uint8)
# arrColorsBGR = arrColors[:, :, ::-1] # RGB to BGR
arrColorsBGR = cv2.cvtColor(arrColors, cv2.COLOR_RGB2BGR)


# Display the image
# cv2.imshow("Scene image", arrColorsBGR)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## ArUco detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(arrColorsBGR)

## Display
# Draw the detected markers on the image
image_with_markers = cv2.aruco.drawDetectedMarkers(arrColorsBGR.copy(), markerCorners, markerIds)

# Display the image with markers
cv2.imshow("Detected ArUco Markers", image_with_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Convert to numpy arrays
markerCorners = np.asarray(markerCorners)
markerIds = np.asarray(markerIds)

# Find the index where marker ID is 0
index_id_0 = np.where(markerIds.flatten() == 0)[0] ## [True, False] --> 0

## Extract corners for marker ID 0
if index_id_0.size > 0:
    corners_id_0 = markerCorners[index_id_0[0]][0]
else:
    corners_id_0 = None

## Getting the 4 corners in camera coordinates [X, Y, Z] [mm]
arrCamPoints = []
for corner in corners_id_0.astype(int):
    pointCam = arrPoints[corner[1], corner[0], :] ## [y, x, :] for array indexing (row, column)

    arrCamPoints.append(pointCam)

arrCamPoints = np.array(arrCamPoints)

## Problem no point in z-direction --> point above origin perpendicular to the plane
oFittedPlane, arrInliersIndex = pcdScene.segment_plane(
        distance_threshold=10,
        ransac_n=100,
        num_iterations=1000,
        probability=0.999
)

pcdSceneNoBackground = pcdScene.select_by_index(arrInliersIndex, invert=True)

pcdBackground = pcdScene.select_by_index(arrInliersIndex) ## Get pointcloud of all points on plane
pcdBackground = pcdBackground.paint_uniform_color([1.0, 0, 0]) ## Paint red
# display_point_clouds([pcdBackground, pcdSceneNoBackground], "Input scene - plane segmenting", False)

vecPlaneNormal = np.array(oFittedPlane[:3])
vecPlaneNormalUnit = vecPlaneNormal / np.linalg.norm(vecPlaneNormal)

point_z = arrCamPoints[1] + 50 * vecPlaneNormalUnit

arrCamPoints[1] = point_z

print("ArUco markers in camera coordinates")
print(arrCamPoints)

arrWorldPoints = np.array([
    [(arrCamPoints[0, 0] - arrCamPoints[3, 0]),
     (arrCamPoints[0, 1] - arrCamPoints[3, 1]),
     0],
    [0, 0, -50],
    [(arrCamPoints[2, 0] - arrCamPoints[3, 0]),
     (arrCamPoints[2, 1] - arrCamPoints[3, 1]),
     0],
    [0, 0, 0]
])

print("ArUco markers in world coordinates")
print(arrWorldPoints)


## Estimate transformation matrix
retval, out, inliers = cv2.estimateAffine3D(arrCamPoints, arrWorldPoints)

## 3x4 to 4x4
trans_mat = np.vstack((out, [0, 0, 0, 1]))

print("Transformation matrix")
print(trans_mat)

pcdSceneFiltered, _ = pcdSceneDown.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)

pcdScene = pcdSceneFiltered

display_point_clouds([pcdScene.transform(trans_mat)], "Scene", False, True, 100)
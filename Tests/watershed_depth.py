import cv2
import open3d as o3d
import numpy as np
from pose_estimation.utils import display_point_clouds
import time
import matplotlib.pyplot as plt


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transforms a set of 3D points using a homogeneous transformation matrix.

    :param points: (N, 3) NumPy array of 3D points.
    :param transform: (4, 4) NumPy array representing the homogeneous transformation matrix.
    :return: (N, 3) NumPy array of transformed 3D points.
    """

    if points.shape[1] != 3:
        raise ValueError("Input points should have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("Transformation matrix should have shape (4, 4)")

    # Convert points to homogeneous coordinates (N, 4)
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])

    # Apply transformation
    transformed_homogeneous = (transform @ homogeneous_points.T).T

    # Convert back to 3D coordinates
    transformed_points = transformed_homogeneous[:, :3]

    return transformed_points


img_with_objects = cv2.imread("./test_input/2025-03-28_14-21-12.jpg")
img_background = cv2.imread("./test_input/2025-03-28_14-46-40.jpg")

pcd = o3d.io.read_point_cloud("./test_input/2025-04-03_11-48-09.ply")


## Transform to cobot coordinate system
twc = np.load("../CalibrationData/tcw.npy")
twc_scale = float(np.load("../CalibrationData/scale_tcw.npy"))

arrPoints = np.asarray(pcd.points)
## Flip to right handed
arrPoints[:, 1] = -arrPoints[:, 1]
arrPoints = transform_points(arrPoints*twc_scale, twc)

pcd_transformed = o3d.geometry.PointCloud()
pcd_transformed.points = o3d.utility.Vector3dVector(arrPoints)
pcd_transformed.colors = o3d.utility.Vector3dVector(pcd.colors)

display_point_clouds([pcd_transformed], "Pointcloud - transformed", False, True, 100)

## Generate mask for 3D bounding box
from matplotlib.path import Path

def is_point_in_quadrilateral(quad, point):
    """
    Determine if a point is inside a quadrilateral.

    :param quad: A list of four (x, y) tuples representing the quadrilateral's vertices.
    :param point: A tuple (x, y) representing the point to check.
    :return: True if the point is inside the quadrilateral, False otherwise.
    """
    quad_path = Path(quad)
    return quad_path.contains_point(point)

## Points from cobot
p1 = (356.331, 434.791)
p2 = (192.801, 607.212)
p3 = (419.131, 826.673)
p4 = (592.422, 665.927)

quad = [p1, p2, p3, p4]


image_width = 1920
image_height = 1080

points = arrPoints.reshape(image_height, image_width, 3)
depth_image = points[:, :, 2]
depth_image[(depth_image < 0) | (depth_image > 200)] = 0
print(depth_image.shape)

# Create mask using is_point_in_quadrilateral
start_time = time.time()
quad_path = Path(quad)
# mask_2d = np.zeros((image_height, image_width), dtype=bool)
# for y in range(image_height):
#     for x in range(image_width):
#         if quad_path.contains_point((points[y, x, 0], points[y, x, 1])):
#             mask_2d[y, x] = True

flat_points = points[:, :, :2].reshape(-1, 2)
print(flat_points.shape)
mask_flat = quad_path.contains_points(flat_points)
mask_2d = mask_flat.reshape(points.shape[:2])
print(f"Mask calculation took: {time.time() - start_time:.2f} seconds")


filtered_points = points[mask_2d == 1]
# Create a new point cloud with the selected points
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries([filtered_pcd])

## Remove bin plane
plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=20,
                                         ransac_n=3,
                                         num_iterations=1000)

filtered_pcd = filtered_pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([filtered_pcd])

## DBSCAN Clustering
labels = np.array(
    filtered_pcd.cluster_dbscan(eps=6, min_points=200)
)

print(labels.shape)

max_label = labels.max()

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
filtered_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([filtered_pcd])

# # Convert mask to uint8 and apply it
# mask_image = np.uint8(mask_2d * 255)
#
# # Apply morphological filtering
# kernel = np.ones((7, 7), np.uint8)  # Kernel for morphological operations
# mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=3)  # Close gaps
#
#
# cv2.imshow("Masked Depth Image", mask_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# normalized_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
# normalized_depth_image = np.uint8(normalized_depth_image)
#
# cv2.imshow("Depth image", normalized_depth_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# ## Apply mask to depth image
# masked_depth_image = cv2.bitwise_and(normalized_depth_image, normalized_depth_image, mask=mask_image)
#
# cv2.imshow("Masked Depth Image", masked_depth_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# ## ------------- Watershed algorithm ----------
# gray = masked_depth_image.copy()
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# kernel = np.ones((7, 7), np.uint8)  # Kernel for morphological operations
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # Remove noise
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Close gaps
#
# cv2.imshow("Mask", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print(np.unique(thresh))
#
# dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
# dist_transform = np.uint8(dist_transform)
# dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
#
# print(np.unique(dist_transform))
#
# cv2.imshow("Distance transform", dist_transform)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# sure_bg = cv2.dilate(thresh, kernel)
# _, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, cv2.THRESH_BINARY)
# sure_fg = np.uint8(sure_fg)
#
# cv2.imshow("Sure foreground", sure_fg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# cv2.imshow("Unknown", unknown)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# ## Create markers
# _, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# ## Mark unknown as 0
# markers[unknown == 255] = 0
# print(markers.shape)
#
#
# print(np.unique(markers))
#
# markers = cv2.watershed(img_with_objects, markers)
# print(np.unique(markers))
# img_with_objects[markers == -1] = [255,0,0]
#
# cv2.imshow("Watershed result", img_with_objects)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

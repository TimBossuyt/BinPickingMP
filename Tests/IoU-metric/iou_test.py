import open3d as o3d
import numpy as np
from shapely.geometry import Polygon

"""
This script was used to test the 2D-IoU metric from the projected bounding boxes of the object and the model
"""

model_mesh = o3d.io.read_triangle_mesh("./model.stl")

pcd = model_mesh.sample_points_poisson_disk(number_of_points=500)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=(0, 0, 0))

# --- Transformation ---
# Rotation: around Z-axis by 30 degrees
R = pcd.get_rotation_matrix_from_xyz((-0.5, 0.5, 0.1))  # (rx, ry, rz)

# Translation: move it along X and Y
T = np.array([0, 0, 20])

# Apply transformation
pcd_transformed = o3d.geometry.PointCloud()
pcd_transformed.points = pcd.points
pcd_transformed.rotate(R, center=(0, 0, 0))
pcd_transformed.translate(T)

o3d.visualization.draw_geometries([pcd, origin, pcd_transformed])


## Calculate bounding box
bbox_1 = np.asarray(pcd.get_oriented_bounding_box().get_box_points())
bbox_2 = np.asarray(pcd_transformed.get_oriented_bounding_box().get_box_points())


## !! order is important (see 2d.py)
xy_1 = [(float(bbox_1[i][0]), float(bbox_1[i][1])) for i in [0, 1, 7, 2]]
xy_2 = [(float(bbox_2[i][0]), float(bbox_2[i][1])) for i in [0, 1, 7, 2]]

# Create shapely Polygons (they'll automatically close the loop)
poly1 = Polygon(xy_1)
poly2 = Polygon(xy_2)

# Compute intersection and union areas
intersection = poly1.intersection(poly2).area
union = poly1.union(poly2).area

# Calculate IoU
iou = intersection / union if union > 0 else 0

print(f"IoU: {iou:.4f}")



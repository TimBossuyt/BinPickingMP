import open3d as o3d
import numpy as np

## 1. Read pointcloud from file
pcd = o3d.io.read_point_cloud("2025-03-01_16-15-08.ply")

## 2. Read .npy file = transformation matrix
matrix = np.load("../camera_calibration_matrix.npy")

## 3. Transform and visualize
pcd.transform(matrix)

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=(0, 0, 0))

o3d.visualization.draw_geometries([pcd, origin])
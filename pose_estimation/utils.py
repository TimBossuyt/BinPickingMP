import numpy as np
import open3d as o3d

def filter_points_by_x_range(pcd, x_min, x_max):
    """
    Removes points from the input point cloud where x values are in the range [x_min, x_max].

    Parameters:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        x_min (float): The minimum x value of the range.
        x_max (float): The maximum x value of the range.

    Returns:
        open3d.geometry.PointCloud: The filtered point cloud.
    """

    # Get the numpy array of points
    points = np.asarray(pcd.points)

    # Apply the mask to get points inside the range [x_min, x_max]
    mask = (points[:, 0] > x_min) & (points[:, 0] < x_max)
    filtered_points = points[mask]

    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Copy colors if available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    # Copy normals if available
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    return filtered_pcd
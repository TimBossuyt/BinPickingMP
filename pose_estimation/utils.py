import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def filter_points_by_x_range(pcd: o3d.geometry.PointCloud, x_min: int, x_max:int) -> o3d.geometry.PointCloud:
    """
    :param pcd: Input point cloud containing 3D points
    :param x_min: Minimum x-coordinate value to filter the points
    :param x_max: Maximum x-coordinate value to filter the points
    :return: A new point cloud containing only the points within the specified x-coordinate range
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

def filter_points_by_z_range(pcd: o3d.geometry.PointCloud, z_min: int, z_max:int) -> o3d.geometry.PointCloud:
    """
    :param pcd: Input point cloud containing 3D points
    :param z_min: Minimum x-coordinate value to filter the points
    :param z_max: Maximum x-coordinate value to filter the points
    :return: A new point cloud containing only the points within the specified x-coordinate range
    """

    # Get the numpy array of points
    points = np.asarray(pcd.points)

    # Apply the mask to get points inside the range [x_min, x_max]
    mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
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

def visualizeDensities(arrDensities: np.ndarray, msh: o3d.geometry.Vertex):
    """
    :param arrDensities: An array containing the density values associated with the vertices of a mesh.
                        The values are used to map corresponding vertex colors using a colormap.
    :param msh: The input mesh object from Open3D containing vertices, triangles, and triangle normals.
    :return: None. Displays a visualization of the mesh with vertex colors mapped based on the provided density values.
    """
    densities = arrDensities
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = msh.vertices
    density_mesh.triangles = msh.triangles
    density_mesh.triangle_normals = msh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    display_point_clouds([density_mesh], "Density mesh visualization", False)


def display_point_clouds(arrPointClouds: [o3d.geometry.PointCloud], sWindowTitle: str,
                         bShowNormals:bool=False, bShowOrigin:bool=False, iOriginSize:int=0):
    """
    :param arrPointClouds: List of open3d.geometry.PointCloud objects to be displayed.
    :param sWindowTitle: Title of the visualization window.
    :param bShowNormals: Boolean indicating whether to display normals for the point clouds.
    :param bShowOrigin: Boolean indicating whether to display the coordinate origin in the visualization.
    :param iOriginSize: Size of the coordinate origin if displayed.
    :return: None
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
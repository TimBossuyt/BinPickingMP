import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

def visualizeDensities(arrDensities, msh):
    """
    Visualizes the vertex densities on a mesh using a color map.

    Parameters:
    - arrDensities: Array of densities for each vertex.
    - msh: Open3D TriangleMesh object.

    Returns:
    None
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
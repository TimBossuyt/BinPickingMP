import open3d as o3d
import numpy as np
import copy

def filter_points_by_z_range(pcd: o3d.geometry.PointCloud, z_min: int, z_max: int) -> o3d.geometry.PointCloud:
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


def create_arrow_on_top(pcd: o3d.geometry.PointCloud, arrow_height=50) -> o3d.geometry.TriangleMesh:
    """
    Create an arrow pointing in the +Z direction, placed just above the top of the point cloud.
    """
    # Find highest Z value
    max_z = np.max(np.asarray(pcd.points)[:, 2])
    center_xy = np.mean(np.asarray(pcd.points)[:, :2], axis=0)

    # Arrow starts just above the top
    arrow_origin = [center_xy[0], center_xy[1], max_z + 5]

    # Create arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=1.5, cone_radius=3.0,
        cylinder_height=arrow_height - 10, cone_height=10
    )
    arrow.paint_uniform_color([0, 0, 1])  # Blue

    # Move arrow to above the point cloud
    arrow.translate(arrow_origin)

    return arrow

## 1. Read CAD-model
model = o3d.io.read_triangle_mesh("./T-stuk-50.stl")
model_pcd = model.sample_points_poisson_disk(number_of_points=2000)
model_pcd_half = filter_points_by_z_range(model.sample_points_poisson_disk(number_of_points=5000), 15, 500)

## Create 2 copies each a separate colour (one on the origin one with an offset)
pcd_origin = model_pcd_half
pcd_offset = copy.deepcopy(model_pcd_half).translate((0, 150, 0), relative=True)

pcd_origin.paint_uniform_color([1, 0, 0])
pcd_offset.paint_uniform_color([1, 0, 0])


# 3. Create coordinate frames
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
offset_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 150, 0])

## Detection pointclouds
pcd1 = copy.deepcopy(model_pcd).translate((0, -1, 0), relative=True)
pcd2 = copy.deepcopy(model_pcd).translate((0, 150, 0), relative=True)

pcd1.paint_uniform_color([0, 1, 0])
pcd2.paint_uniform_color([0, 1, 0])

arrow1 = create_arrow_on_top(pcd1)
arrow2 = create_arrow_on_top(pcd2)

arrow1.paint_uniform_color([1, 192/255, 203/255])
arrow2.paint_uniform_color([1, 192/255, 203/255])



R = np.array([[1,  0,  0],
              [0, -1,  0],
              [0,  0, -1]])

pcd1.rotate(R, center=pcd1.get_center())
arrow1.rotate(R, center=arrow1.get_center())


## Plot
o3d.visualization.draw_geometries([pcd_origin, pcd_offset, origin_frame, offset_frame, pcd1, pcd2, arrow1, arrow2])
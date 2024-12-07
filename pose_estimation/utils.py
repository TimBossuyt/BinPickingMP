import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def extract_translation_rotation(transform_matrix):
    # Extract the translation part (x, y, z) from the 4th column of the matrix
    translation = transform_matrix[:3, 3]
    x, y, z = translation[0], translation[1], translation[2]

    # Extract the rotation matrix (upper-left 3x3 sub-matrix)
    rotation_matrix = transform_matrix[:3, :3]

    # Use scipy to get Euler angles from the rotation matrix (in degrees)
    rotation = R.from_matrix(rotation_matrix)
    anglex, angley, anglez = rotation.as_euler('xyz', degrees=True)

    # Print the results nicely to the terminal
    print("Translation (x, y, z):")
    print(f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}")

    print("\nRotation angles (degrees):")
    print(f"Angle around x-axis: {anglex:.2f}°")
    print(f"Angle around y-axis: {angley:.2f}°")
    print(f"Angle around z-axis: {anglez:.2f}°")

    return x, y, z, anglex, angley, anglez

def mean_to_mean_transformation(pcdModel, pcdScene):
    arrScenePoints = np.asarray(pcdScene.points)
    arrModelPoints = np.asarray(pcdModel.points)

    arrMeanScene = np.mean(arrScenePoints, axis=0)
    arrMeanModel = np.mean(arrModelPoints, axis=0)

    arrInitTranslation = arrMeanScene - arrMeanModel

    transformation_init = np.eye(4)
    transformation_init[:3, 3] = arrInitTranslation

    return transformation_init

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


def planeSegmenting(pcd, iDistanceThreshold, iRansacN=100, iIterations=1000, iProb=0.9999):
    """
    Segments a plane from a point cloud using RANSAC.

    Parameters:
    - pcd: Open3D PointCloud object.
    - iDistanceThreshold: Maximum distance a point can be from the plane to be considered an inlier.
    - iRansacN: Number of points sampled to fit the plane.
    - iIterations: Maximum number of RANSAC iterations.
    - iProb: Probability of success for RANSAC.

    Returns:
    - arrInliersIndex: Indices of the inlier points forming the plane.
    """
    oFittedPlane, arrInliersIndex = pcd.segment_plane(
        distance_threshold=iDistanceThreshold,
        ransac_n=iRansacN,
        num_iterations=iIterations,
        probability=iProb
    )

    return arrInliersIndex


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


def surfaceReconstructionROI(pcdROI, bVisualize,
                             iRawNormalRadius, iPoissonDepth,
                             iDensityThreshold, iTaubinIter, iPoints):
    """
    Reconstructs a surface from a region of interest in a point cloud.

    Parameters:
    - pcdROI: Input point cloud (region of interest).
    - bVisualize: Show density visualization if True.
    - iRawNormalRadius: Radius for normal estimation.
    - iPoissonDepth: Depth parameter for Poisson surface reconstruction.
    - iDensityThreshold: Quantile threshold for density-based vertex removal.
    - iTaubinIter: Number of iterations for Taubin smoothing.
    - iPoints: Number of points to sample in the output.

    Returns:
    - pcdROIReconstructed: A reconstructed pointcloud with normals
    """

    ## Preparation (normal estimation)
    pcdROI.estimate_normals()
    pcdROI.orient_normals_to_align_with_direction([0, 0, 1])
    oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=iRawNormalRadius)

    pcdROI.estimate_normals(oNormalSearchParam)

    ## Poisson surface reconstruction (based on normals)
    mshSurfRec, arrDensities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd=pcdROI,
        depth=iPoissonDepth
    )

    arrDensities = np.asarray(arrDensities)
    if bVisualize:
        visualizeDensities(arrDensities, mshSurfRec)

    ## Removing points with low density
    arrVerticesToRemove = arrDensities < np.quantile(arrDensities, iDensityThreshold)
    mshSurfRec.remove_vertices_by_mask(arrVerticesToRemove)

    ## Smoothing with Taubin filter
    mshSurfRecSmooth = mshSurfRec.filter_smooth_taubin(number_of_iterations=iTaubinIter)
    mshSurfRecSmooth.compute_vertex_normals()

    pcdROIReconstructed = mshSurfRecSmooth.sample_points_poisson_disk(number_of_points=iPoints)

    return pcdROIReconstructed
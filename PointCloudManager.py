import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from ExtrinsicTransformationAruco import ExtrinsicTransformationAruco
import cv2


## Utility functions
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


class PointCloudManager:
    """
    Manages loading, preprocessing, clustering, and reconstructing point clouds.
    """

    def __init__(self):
        self.pcdScene = None
        self.pcdModel = None

        self.pcdSceneNoBackground = None

        self.oTransformation = None


        self.arrPcdObjectsRaw = []
        self.arrPcdObjectsReconstructed = []

    def loadModelMesh(self, sModelPath, iPoints):
        """
        Loads a 3D model as a mesh and samples it into a point cloud.

        Parameters:
        - sModelPath: Path to the model file.
        - iPoints: Number of points to sample.
        """

        mshModel = o3d.io.read_triangle_mesh(sModelPath)

        self.pcdModel = mshModel.sample_points_poisson_disk(number_of_points=iPoints)

    def loadScenePointCloud(self, sScenePath, iVoxelSize, iOutlierNeighbours, iStd, iROIX, iROIY):
        """
       Loads and preprocesses a scene point cloud.

       Parameters:
       - sScenePath: Path to the point cloud file.
       - iVoxelSize: Voxel size for downsampling.
       - iOutlierNeighbours: Minimum neighbors for statistical outlier removal.
       - iStd: Standard deviation multiplier for outlier removal.
       """

        pcdSceneRaw = o3d.io.read_point_cloud(sScenePath)

        ##### Basic preprocessing
        ## Transform to World Coordinates
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.oTransformation = ExtrinsicTransformationAruco(pcdSceneRaw, 1080, 1920, aruco_dict)

        pcdSceneRaw = pcdSceneRaw.transform(self.oTransformation.arrTransMat)

        ## Voxel down
        pcdSceneDown = pcdSceneRaw.voxel_down_sample(voxel_size=iVoxelSize)

        ## Remove outliers
        pcdSceneFiltered, _ = pcdSceneDown.remove_statistical_outlier(nb_neighbors=iOutlierNeighbours, std_ratio=iStd)

        self.pcdScene = pcdSceneFiltered

        ## Select ROI
        min_bound = [0, 0, -float("inf")]
        max_bound = [iROIX, iROIY, float("inf")]

        # Create an Axis-Aligned Bounding Box
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        self.pcdScene = self.pcdScene.crop(aabb)


    def displayScene(self):
        """Displays the scene point cloud."""
        display_point_clouds([self.pcdScene], "Scene", False, True, 100)

    def displayModel(self):
        """Displays the model point cloud."""
        display_point_clouds([self.pcdModel], "Model", False, True, 20)

    def displayClusters(self):
        """Displays the clustered objects with random colors."""
        arrColoredPcds = []
        for pcd in self.arrPcdObjectsRaw:
            ## Generate random color
            oRandomColor = np.random.rand(3)

            ## Assign color to all points in pointcloud
            pcd.colors = o3d.utility.Vector3dVector(np.tile(oRandomColor, (np.asarray(pcd.points).shape[0], 1)))

            arrColoredPcds.append(pcd)

        display_point_clouds(arrColoredPcds, "Clusters", False)

    def displayReconstructedObjects(self):
        """Displays reconstructed objects."""
        display_point_clouds(self.arrPcdObjectsReconstructed, "Reconstructed objects", True, True, 100)

    def displayNoBackgroundScene(self):
        """Displays reconstructed objects."""
        display_point_clouds([self.pcdSceneNoBackground], "Scene - no background", False, True, 100)

    def removeSceneBackgroundPlane(self, iPlaneDistance):
        """
        Removes the dominant plane from the scene using RANSAC.

        Parameters:
        - iPlaneDistance: Maximum distance for a point to be considered on the plane.
        """
        arrPlaneSegmentIndex = planeSegmenting(
            pcd=self.pcdScene,
            iDistanceThreshold=iPlaneDistance,
        )

        self.pcdSceneNoBackground = self.pcdScene.select_by_index(arrPlaneSegmentIndex, invert=True)

    def clusterScene(self, iEpsilon, iMinPointsDensity, iPointsValidObject):
        """
        Clusters the scene into objects using DBSCAN.

        Parameters:
        - iEpsilon: Neighborhood radius for clustering.
        - iMinPointsDensity: Minimum points required for a cluster.
        - iPointsValidObject: Minimum points for a valid object cluster.
        """

        self.arrPcdObjectsRaw = []

        arrLabels = np.array(self.pcdSceneNoBackground.cluster_dbscan(
            eps=iEpsilon,
            min_points=iMinPointsDensity
        ))

        ## Saving largest clusters
        for label in set(arrLabels):
            if label != -1:
                arrPoints = np.asarray(self.pcdSceneNoBackground.points)[arrLabels == label]
                if len(arrPoints) >= iPointsValidObject:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(arrPoints)

                    self.arrPcdObjectsRaw.append(pcd)

        print(f"Saved {len(self.arrPcdObjectsRaw)} cluster(s)")

    def surfaceReconstructionObjects(self, bVisualize, iRawNormalRadius, iPoissonDepth,
                                     iDensityThreshold, iTaubinIter, iPoints):

        """
        Performs surface reconstruction on all clustered objects in the scene.

        Parameters:
        - bVisualize: Boolean indicating whether to visualize vertex densities.
        - iRawNormalRadius: Radius used for normal estimation during reconstruction.
        - iPoissonDepth: Depth parameter for Poisson surface reconstruction.
        - iDensityThreshold: Quantile threshold to remove low-density vertices.
        - iTaubinIter: Number of iterations for Taubin smoothing to improve surface quality.
        - iPoints: Number of points to sample from each reconstructed surface.
        Updates the `arrPcdObjectsReconstructed` list with reconstructed objects.
        """

        for pcdObject in self.arrPcdObjectsRaw:
            pcdReconstructed = surfaceReconstructionROI(
                pcdROI=pcdObject,
                bVisualize=bVisualize,
                iRawNormalRadius=iRawNormalRadius,
                iPoissonDepth=iPoissonDepth,
                iDensityThreshold=iDensityThreshold,
                iTaubinIter=iTaubinIter,
                iPoints=iPoints
            )

            self.arrPcdObjectsReconstructed.append(pcdReconstructed)

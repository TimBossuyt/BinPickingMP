import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from .utils import *


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

    # Apply the mask to filter points outside the range [x_min, x_max]
    mask = (points[:, 0] < x_min) | (points[:, 0] > x_max)
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

        ## TO REMOVE!!!!
        self.pcdModel = filter_points_by_x_range(self.pcdModel, -5, 5)


    def loadScenePointCloud(self, sScenePath, iVoxelSize, iOutlierNeighbours, iStd, arrWCTransform, iROIX, iROIY):
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
        pcdSceneRaw = pcdSceneRaw.transform(arrWCTransform)

        ## Voxel down
        pcdSceneDown = pcdSceneRaw.voxel_down_sample(voxel_size=iVoxelSize)

        ## Remove outliers
        pcdSceneFiltered, _ = pcdSceneDown.remove_statistical_outlier(nb_neighbors=iOutlierNeighbours, std_ratio=iStd)

        self.pcdScene = pcdSceneFiltered

        ## Select ROI
        min_bound = [0, 0, -50]
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

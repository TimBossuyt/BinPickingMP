import open3d as o3d
from .utils import *

class Scene:
    """
    Manages scene pointcloud for pose estimation
    """
    def __init__(self, sScenePath, arrWCTransform, iVoxelSize, iOutlierNeighbours, iStd):
        ## Initializing
        self.arrPcdObjectsRaw = []
        self.arrPcdObjectsReconstructed = []
        self.pcdSceneNoBackground = None

        ## Load camera extrinsics
        self.extrinsics = arrWCTransform

        ## Read raw point cloud data from file
        self.pcdSceneRaw = o3d.io.read_point_cloud(sScenePath)

        #### Basic preprocessing
        ## Voxel down
        self.pcdSceneDown = self.pcdSceneRaw.voxel_down_sample(voxel_size=iVoxelSize)

        ## Filter and remove outliers
        self.pcdScene, _ = self.pcdSceneDown.remove_statistical_outlier(
            nb_neighbors=iOutlierNeighbours, std_ratio=iStd
        )


    def transformAndCrop_(self, iXmin, iXmax, iYmin, iYmax):
        self.pcdScene = self.pcdScene.transform(self.extrinsics)

        ## Select ROI (From world coordinate frame)
        min_bound = [iXmin, iYmin, -50]
        max_bound = [iXmax, iYmax, float("inf")]

        ## Create Axis-Aligned Bounding Box
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )

        self.pcdScene = self.pcdScene.crop(aabb)


    def clusterScene_(self, iEpsilon, iMinPointsDensity, iPointsValidObject):
        """
        Clusters the scene (with already removed background) into objects using DBSCAN.

        Parameters:
        - iEpsilon: Neighborhood radius for clustering.
        - iMinPointsDensity: Minimum points required for a cluster.
        - iPointsValidObject: Minimum points for a valid object cluster.
        """

        arrLabels = np.array(self.pcdSceneNoBackground.cluster_dbscan(
            eps=iEpsilon,
            min_points=iMinPointsDensity
        ))

        ## Saving largest clusters
        for label in set(arrLabels):
            ## If label not -1 (= Noise)
            if label != -1:
                ## Get points that are assigned to this label
                arrPoints = np.asarray(self.pcdSceneNoBackground.points)[arrLabels == label]

                ## Check if cluster is large enough to form an object
                if len(arrPoints) >= iPointsValidObject:
                    ## Create Open3D pointcloud from points
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(arrPoints)

                    ## Add raw pointcloud clusters to list
                    self.arrPcdObjectsRaw.append(pcd)

        # print(f"Saved {len(self.arrPcdObjectsRaw)} cluster(s)")


    def removeSceneBackgroundPlane_(self, iPlaneDistance):
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

    def surfaceReconstructionObjects_(self, bVisualize, iRawNormalRadius, iPoissonDepth,
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


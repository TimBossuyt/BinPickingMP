import open3d as o3d
import matplotlib.path as mplpath
from scipy.spatial import ConvexHull
import numpy as np
import random
import logging
import time

## ---------- Custom imports ----------
from .segmentation import ObjectSegmentation
from .settings import SettingsManager
from .utils import visualizeDensities, display_point_clouds
## ------------------------------------

logger = logging.getLogger("Scene")

class Scene:
    """
    A class for handling 3D point cloud data, performing object segmentation, processing detected objects,
    and reconstructing surfaces.
    """

    def __init__(self, raw_pcd: o3d.geometry.PointCloud, settingsmanager: SettingsManager, samModel):
        ## Set and load the settings manager
        self.oSm = settingsmanager
        self._loadSettings()

        ## Load pointcloud
        self.pcdRaw = raw_pcd

        ## Create image (BGR) from pointcloud data
        self.arrColours = np.asarray(self.pcdRaw.colors).reshape(self.iHeightImage, self.iWidthImage, 3) * 255
        self.arrColours = np.uint8(self.arrColours)

        ## Save the ROI of the scene
        self.pcdROI = self._selectROI()
        self.pcdViz = self._cropViz()

        ## Initialize object segmentation
        oSegmentation = ObjectSegmentation(
            oSettingsManager=self.oSm,
            model=samModel
        )

        self.objectMasks = oSegmentation.getMasksFromImage(self.arrColours)

        logger.info("Applying masks to pointcloud")
        tStart = time.time()
        ## Create a dictionary with points of each object (using clustering algorithm)
        self.dictObjects = self._createObjectsDict()

        tEnd = time.time()
        logger.info(f"Applying object masks took {(tEnd - tStart)*1000:.2f} ms")

        # ## Process the object point clouds
        self.dictProcessedPcds = self._processObjects()



    def _loadSettings(self) -> None:
        ## --------------- ROI Settings ---------------
        self.ptRoi1 = self.oSm.get("Scene.ROI.p1")
        self.ptRoi2 = self.oSm.get("Scene.ROI.p2")
        self.ptRoi3 = self.oSm.get("Scene.ROI.p3")
        self.ptRoi4 = self.oSm.get("Scene.ROI.p4")

        self.iBinPlaneDistance = self.oSm.get("Scene.ROI.BinPlaneDistance")

        ## --------------- Clustering ---------------
        self.clustEpsilon = self.oSm.get("Scene.Clustering.Epsilon")
        self.clustMinPoints = self.oSm.get("Scene.Clustering.min_points")

        self.minObjectSize = self.oSm.get("Scene.Clustering.MinObjectSize")

        ## --------------- Resolution Settings ---------------
        self.iHeightImage = self.oSm.get("Scene.CameraResolution.HeightImage")
        self.iWidthImage = self.oSm.get("Scene.CameraResolution.WidthImage")

        ## --------------- Basic Processing ---------------
        self.iVoxelSize = self.oSm.get("Scene.BasicProcessing.VoxelSize")
        self.iOutlierNeighbours = self.oSm.get("Scene.BasicProcessing.OutlierNeighbours")
        self.iStd = self.oSm.get("Scene.BasicProcessing.Std")

        ## --------------- Surface Reconstruction ---------------
        self.iRawNormalRadius = self.oSm.get("Scene.SurfaceReconstruction.RawNormalRadius")
        self.iProcessedNormalRadius = self.oSm.get("Scene.SurfaceReconstruction.ProcessedNormalRadius")
        self.iPoissonDepth = self.oSm.get("Scene.SurfaceReconstruction.PoissonDepth")
        self.iDensityThreshold = self.oSm.get("Scene.SurfaceReconstruction.DensityThreshold")
        self.iTaubinIter = self.oSm.get("Scene.SurfaceReconstruction.TaubinIter")
        self.iPoints = self.oSm.get("Scene.SurfaceReconstruction.NumberOfPoints")

        self.iOutlierReconstructed = self.oSm.get("Scene.SurfaceReconstruction.ReconstructedOutlierNeighbours")
        self.iStdReconstructed = self.oSm.get("Scene.SurfaceReconstruction.ReconstructedOutlierStd")

        value = self.oSm.get("Scene.SurfaceReconstruction.bVisualize")
        self.bVisualize = (value == 1) ## Sets true if 1 and false if 0

        logger.debug("Settings set correctly")


    def _selectROI(self):
        """
        Returns ROI of pointcloud
        :return:
        """

        ## Create quadrilateral Path (matplotlib)
        quad_points = [self.ptRoi1, self.ptRoi2, self.ptRoi3, self.ptRoi4]
        quad_path = mplpath.Path(quad_points)

        points = np.asarray(self.pcdRaw.points).reshape(self.iHeightImage, self.iWidthImage, 3)
        flat_points = points[:, :, :2].reshape(-1, 2)

        mask_flat = quad_path.contains_points(flat_points)
        mask_2d = mask_flat.reshape(points.shape[:2])

        ## Only get points within x, y region
        filtered_points = points[mask_2d == 1]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # o3d.visualization.draw_geometries([filtered_pcd], window_name="Scene - ROI")

        ## Remove bin plane
        plane_model, inliers = filtered_pcd.segment_plane(
            distance_threshold=self.iBinPlaneDistance,
            ransac_n = 3,
            num_iterations=1000,
        )
        ## Only select outliers (not plane) as part of ROI
        filtered_pcd = filtered_pcd.select_by_index(inliers, invert=True)
        # o3d.visualization.draw_geometries([filtered_pcd], window_name="Scene - ROI - Removed Plane")
        if self.bVisualize:
            display_point_clouds([filtered_pcd], "Scene - ROI", False, True, 100)

        return filtered_pcd

    def _cropViz(self):
        """
        Returns pointcloud with only interesting points for visualization
        :return:
        """

        ## Add margin to the points
        quad_points = [self.ptRoi1, self.ptRoi2, self.ptRoi3, self.ptRoi4]
        quad_points = np.array(quad_points)

        hull = ConvexHull(quad_points)
        hull_points = quad_points[hull.vertices]

        # Adding a margin of 20 to the convex hull
        margin = 100
        center = np.mean(hull_points, axis=0)  # Center of the hull

        expanded_hull_points = []
        for point in hull_points:
            # Direction vector from center to the point
            direction = point - center
            # Normalize the direction and expand by margin
            expanded_point = point + direction / np.linalg.norm(direction) * margin
            expanded_hull_points.append(expanded_point)
        expanded_hull_points = np.array(expanded_hull_points)
        quad_path = mplpath.Path(expanded_hull_points)

        points = np.asarray(self.pcdRaw.points).reshape(self.iHeightImage, self.iWidthImage, 3)
        flat_points = points[:, :, :2].reshape(-1, 2)

        mask_flat = quad_path.contains_points(flat_points)
        mask_2d = mask_flat.reshape(points.shape[:2])

        # Only get points within the x, y region and z coordinate between 0 and 200
        z_mask = (points[:, :, 2] >= 0) & (points[:, :, 2] <= 200)  # Apply Z condition
        combined_mask = mask_2d & z_mask  # Combine x, y mask with z condition

        ## Only get points within x, y region
        filtered_points = points[combined_mask == 1]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        ## Apply colours
        colours = np.asarray(self.pcdRaw.colors).reshape(self.iHeightImage, self.iWidthImage, 3)
        # print(colours.shape)

        filtered_colours = colours[combined_mask == 1]
        # print(filtered_colours.shape)
        # filtered_colours = cv2.cvtColor(filtered_colours, cv2.COLOR_BGR2RGB) / 255
        ## Swap BGR to RGB
        filtered_colours = filtered_colours[..., ::-1]

        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colours)

        # o3d.visualization.draw_geometries([filtered_pcd], window_name="Scene - Viz")
        return filtered_pcd

    def displayObjectPoints(self) -> None:
        """
        Generates a 3D visualization of processed point clouds where each object is displayed with a unique color.
        Adds a coordinate frame as the origin to the visualization.

        :return: None
        """
        geometries = []
        ## 3D Plot of each object with different colors
        for _id, pointcloud in self.dictProcessedPcds.items():
            # Generate a random color for the object
            color = [random.random(), random.random(), random.random()]
            pointcloud.paint_uniform_color(color)

            geometries.append(pointcloud)

        ## Add origin to visualization
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        geometries.append(origin)
        geometries.append(self.pcdROI)

        o3d.visualization.draw_geometries(geometries)

    def _createObjectsDict(self) -> dict[int, np.ndarray]:
        """
        Creates a dictionary of objects where each key corresponds to an object's ID and the value
        is a numpy array of XYZ coordinates associated with that object.

        :return: Dictionary where the keys are object IDs (integers) and the values are numpy arrays of XYZ points.
        """
        dictObjects = {}

        arrPoints = np.asarray(self.pcdRaw.points).reshape(self.iHeightImage, self.iWidthImage, 3)

        for i, object_mask in enumerate(self.objectMasks):
            print(f"Creating points collection for object {i}")

            points = arrPoints[object_mask==True]

            ## --- DEBUGGING -----
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if self.bVisualize:
                display_point_clouds([pcd], "Apply mask: Object " + str(i), False, True, 100)
            ## -------------------

            dictObjects[i] = points

        return dictObjects

    def _processObjects(self) -> dict[int, o3d.geometry.PointCloud]:
        """
        Processes objects by converting their points into PointCloud objects and mapping them to their respective IDs.

        :return: A dictionary mapping object IDs (int) to their corresponding PointCloud objects (o3d.geometry.PointCloud)
        """
        pcds = {}

        tStartAll = time.time()

        for _id, points in self.dictObjects.items():
            tStart = time.time()
            pcds[_id] = self._processPoints(points)
            tEnd = time.time()
            logger.debug(f"Processing objects {_id} took {(tEnd - tStart) * 1000:.2f} ms")

        tEndAll = time.time()

        logger.info(f"Processing all objects took {(tEndAll - tStartAll)*1000:.2f} ms")

        return pcds

    def _processPoints(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Processes a given point cloud by performing a series of operations including down-sampling,
        outlier removal, and surface reconstruction.

        :param points: Input point cloud data as a numpy array.
        :type points: np.ndarray
        :return: Processed point cloud with computed normals.
        :rtype: o3d.geometry.PointCloud
        """

        # logger.debug("Starting pointcloud processing")

        ## 1. Create pointcloud from points
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(points)

        ## 2. Voxel down and remove outliers
        pcd_down = pcd_raw.voxel_down_sample(voxel_size=self.iVoxelSize)
        pcd_down, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=self.iOutlierNeighbours,
            std_ratio=self.iStd)

        ## 3. Perform surface reconstruction
        # logger.debug("Performing surface reconstruction")
        pcd_reconstructed = self._surfaceReconstruction(
            pointcloud=pcd_down,
        )

        return pcd_reconstructed


    def _surfaceReconstruction(self, pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        ## 1. First normal estimation
        # Orient normals to camera (upwards)
        # logger.debug("Initial pointcloud normal estimation")
        pointcloud.estimate_normals()
        pointcloud.orient_normals_to_align_with_direction([0, 0, 1])

        # Recalculate normals with search parameters
        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iRawNormalRadius)
        pointcloud.estimate_normals(oNormalSearchParam)

        if self.bVisualize:
            display_point_clouds(arrPointClouds=[pointcloud],
                                 sWindowTitle="Estimated normals before reconstruction",
                                 bShowOrigin=True,
                                 iOriginSize=100
            )

        pointcloud, _ = pointcloud.remove_statistical_outlier(
            nb_neighbors=self.iOutlierReconstructed,
            std_ratio=self.iStdReconstructed)

        ## 2. Perform poisson surface reconstruction
        mshSurfRec, arrDensities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd=pointcloud,
            depth=self.iPoissonDepth
        )

        arrDensities = np.asarray(arrDensities)

        if self.bVisualize:
            visualizeDensities(arrDensities, mshSurfRec)

        ## 3. Remove points with the lowest density
        arrVerticesToRemove = arrDensities < np.quantile(arrDensities, self.iDensityThreshold)
        mshSurfRec.remove_vertices_by_mask(arrVerticesToRemove)

        if self.bVisualize:
            o3d.visualization.draw_geometries([mshSurfRec], window_name="Created mesh after "
                                                                        "removing low density surfaces",
                                              width=800, height=600)

        ## 4. Smoothing with taubin filter
        mshSurfRecSmooth = mshSurfRec.filter_smooth_taubin(number_of_iterations=self.iTaubinIter)
        mshSurfRecSmooth.compute_vertex_normals()

        if self.bVisualize:
            o3d.visualization.draw_geometries([mshSurfRecSmooth], window_name="Mesh after smoothing",
                                              width=800, height=600)

        pointcloud_reconstructed = mshSurfRecSmooth.sample_points_poisson_disk(number_of_points=self.iPoints)

        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iProcessedNormalRadius)
        pointcloud_reconstructed.estimate_normals(oNormalSearchParam)

        if self.bVisualize:
            display_point_clouds(arrPointClouds=[pointcloud_reconstructed],
                                 sWindowTitle="Final reconstructed pointcloud",
                                 bShowOrigin=True,
                                 iOriginSize=100
                                 )

        return pointcloud_reconstructed

    def reload_settings(self):
        self._loadSettings()
        logger.info("Reloaded settings")

import cv2
import open3d as o3d
from pathlib import Path
import numpy as np
import random
import logging

## ---------- Custom imports ----------
from .object_masks import ObjectMasks
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

    def __init__(self, raw_pcd: o3d.geometry.PointCloud, settingsmanager: SettingsManager):
        ## Set and load the settings manager
        self.oSm = settingsmanager
        self.__loadSettings()

        ## Load pointcloud
        self.pcdRaw = raw_pcd

        ## Create image (BGR) from pointcloud data
        self.arrColours = np.asarray(self.pcdRaw.colors).reshape(self.iHeightImage, self.iWidthImage, 3) * 255
        self.arrColours = np.uint8(self.arrColours)

        ## Initialize the object segmentation object
        oSegmentation = ObjectSegmentation(
            oSettingsManager=self.oSm
        )

        self.oMasks = ObjectMasks(self.arrColours, oSegmentation)

        ## Save the masks for each objetc
        self.dictMasks = self.oMasks.getMasks()

        ## 3D Points (camera coordinates)
        self.arrPoints = np.asarray(self.pcdRaw.points).reshape(self.iHeightImage, self.iWidthImage, 3)

        ## Create a dictionary with points of each object
        self.dictObjects = self.__createObjectsDict()

        ## Process the object point clouds
        self.dictProcessedPcds = self.__processObjects()

        ## Save the ROI of the scene
        self.pcdROI = self.__selectROI()

    def __loadSettings(self) -> None:
        ## --------------- ROI Settings ---------------
        self.x_min = self.oSm.get("ObjectSegmentation.ROI.xMin")
        self.y_min = self.oSm.get("ObjectSegmentation.ROI.yMin")
        self.x_max = self.oSm.get("ObjectSegmentation.ROI.xMax")
        self.y_max = self.oSm.get("ObjectSegmentation.ROI.yMax")

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

        value = self.oSm.get("Scene.SurfaceReconstruction.bVisualize")
        self.bVisualize = (value == 1) ## Sets true if 1 and false if 0

        logger.debug("Settings set correctly")


    def __selectROI(self):
        ## Create points and colours arrays with correct shapes
        points = np.asarray(self.pcdRaw.points).reshape(self.iHeightImage, self.iWidthImage, 3)
        colors = np.asarray(self.pcdRaw.colors).reshape(self.iHeightImage, self.iWidthImage, 3)

        sub_points = points[self.y_min:self.y_max, self.x_min:self.x_max]
        sub_colors = colors[self.y_min:self.y_max, self.x_min:self.x_max]

        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(sub_points.reshape(-1, 3))
        sub_pcd.colors = o3d.utility.Vector3dVector(sub_colors.reshape(-1, 3))

        ## Downsample
        sub_pcd_down = sub_pcd.voxel_down_sample(voxel_size=5)

        ## Filter the outliers
        pcd_roi, _ = sub_pcd_down.remove_statistical_outlier(50, 0.1)

        return pcd_roi

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

        o3d.visualization.draw_geometries(geometries)

    def __createObjectsDict(self) -> dict[int, np.ndarray]:
        """
        Creates a dictionary of objects where each key corresponds to an object's ID and the value
        is a numpy array of XYZ coordinates associated with that object.

        For each detected object ID, applies a mask to extract the corresponding XYZ points.
        Points with a z-coordinate value of 0 (invalid points) are removed.

        :return: Dictionary where the keys are object IDs (integers) and the values are numpy arrays of XYZ points.
        """
        dictObjects = {}

        ## apply mask for each detected object to the dictionary with the XYZ points as an array to corresponding key
        for _id, mask in self.dictMasks.items():
            _id = int(_id)
            dictObjects[_id] = self.arrPoints[mask==1]
            ## Remove all points with z = 0
            dictObjects[_id] = dictObjects[_id][dictObjects[_id][:,2]!=0]

        return dictObjects

    def __processObjects(self) -> dict[int, o3d.geometry.PointCloud]:
        """
        Processes objects by converting their points into PointCloud objects and mapping them to their respective IDs.

        :return: A dictionary mapping object IDs (int) to their corresponding PointCloud objects (o3d.geometry.PointCloud)
        """
        pcds = {}

        for _id, points in self.dictObjects.items():
            pcds[_id] = self.__processPoints(points)

        return pcds

    def __processPoints(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Processes a given point cloud by performing a series of operations including down-sampling,
        outlier removal, and surface reconstruction.

        :param points: Input point cloud data as a numpy array.
        :type points: np.ndarray
        :return: Processed point cloud with computed normals.
        :rtype: o3d.geometry.PointCloud
        """

        ## 1. Create pointcloud from points
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(points)

        ## 2. Voxel down and remove outliers
        pcd_down = pcd_raw.voxel_down_sample(voxel_size=self.iVoxelSize)
        pcd_down, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=self.iOutlierNeighbours,
            std_ratio=self.iStd)

        ## 3. Perform surface reconstruction
        # TODO: Remove hardcoded parameters
        pcd_reconstructed = self.__surfaceReconstruction(
            pointcloud=pcd_down,
        )

        return pcd_reconstructed


    def __surfaceReconstruction(self, pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        ## 1. First normal estimation
        # Orient normals to camera (upwards)
        pointcloud.estimate_normals()
        pointcloud.orient_normals_to_align_with_direction([0, 0, -1])

        # Recalculate normals with search parameters
        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iRawNormalRadius)
        pointcloud.estimate_normals(oNormalSearchParam)

        if self.bVisualize:
            display_point_clouds(arrPointClouds=[pointcloud],
                                 sWindowTitle="Estimated normals before reconstruction",
                                 bShowOrigin=True,
                                 iOriginSize=100
            )

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

        ## 4. Smoothing with taubin filter
        mshSurfRecSmooth = mshSurfRec.filter_smooth_taubin(number_of_iterations=self.iTaubinIter)
        mshSurfRecSmooth.compute_vertex_normals()

        pointcloud_reconstructed = mshSurfRecSmooth.sample_points_poisson_disk(number_of_points=self.iPoints)

        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iProcessedNormalRadius)
        pointcloud_reconstructed.estimate_normals(oNormalSearchParam)

        return pointcloud_reconstructed



if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(Path("test_input/2025-02-20_19-46-58.ply"))

    sm = SettingsManager("test_input/default_settings.json")

    oScene = Scene(raw_pcd=pcd,
                   settingsmanager=sm)

    oScene.oMasks.debugSegmentation()

    oScene.displayObjectPoints()
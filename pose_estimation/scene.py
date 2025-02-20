import cv2
import open3d as o3d
from pathlib import Path
import numpy as np
import random

## ---------- Custom imports ----------
from object_masks import ObjectMasks
from segmentation import ObjectSegmentation
from utils import visualizeDensities, display_point_clouds
## ------------------------------------

class SceneParameters:
    def __init__(self):
        ## Define all the default values
        pass

    def loadSettingsJson(self, sFileName):
        pass

    def parseJson(self):
        pass


class Scene:
    """
    A class for handling 3D point cloud data, performing object segmentation, processing detected objects, and reconstructing surfaces.

    Attributes:
    raw_pcd (o3d.geometry.PointCloud): The raw point cloud containing 3D coordinates and color information.
    iWidthImage (int): The width of the corresponding 2D image.
    iHeightImage (int): The height of the corresponding 2D image.
    iVoxelSize (int): The voxel size used for downsampling the point cloud.
    iOutlierNeighbours (int): Number of neighbors to consider when removing statistical outliers.
    iStd (float): The standard deviation threshold for statistical outlier removal.
    arrColours (np.ndarray): The color information in BGR format, reshaped to match the dimensions of the 2D image.
    oMasks (ObjectMasks): Object mask segmentation object.
    dictMasks (dict): A dictionary containing masks for each segmented object.
    arrPoints (np.ndarray): 3D camera coordinates of the points in the cloud.
    dictObjects (dict): A dictionary mapping object IDs to their corresponding 3D points.
    dictProcessedPcds (dict): A dictionary containing processed point clouds for each detected object.

    Methods:
    __init__: Initializes the scene object from a raw point cloud and performs preprocessing including object segmentation.
    displayObjectPoints: Visualizes 3D point clouds for each segmented object with different colors in a 3D space.
    __createObjectsDict: Creates a dictionary mapping object IDs to corresponding 3D points based on segmentation masks.
    __processObjects: Processes each object's 3D points to produce downsampled and reconstructed point clouds.
    __processPoints: Processes a set of 3D points to create a downsampled point cloud, remove outliers, and perform surface reconstruction.
    __surfaceReconstruction: Performs surface reconstruction using Poisson reconstruction, density filtering, and Taubin smoothing.
    """

    def __init__(self, raw_pcd: o3d.geometry.PointCloud, iWidthImage: int, iHeightImage: int,
                 iVoxelSize: int, iOutlierNeighbours: int, iStd: float):

        ## Save settings
        self.pcdRaw = raw_pcd
        self.iVoxelSize = iVoxelSize
        self.iOutlierNeighbours = iOutlierNeighbours
        self.iStd = iStd

        ## Create image (BGR) from pointcloud data
        self.arrColours = np.asarray(self.pcdRaw.colors).reshape(iHeightImage, iWidthImage, 3) * 255
        self.arrColours = np.uint8(self.arrColours)

        ## Initialize the object segmentation object
        oSegmentation = ObjectSegmentation(500, 100, 1500, 800)
        self.oMasks = ObjectMasks(self.arrColours, oSegmentation)

        ## Save the masks for each object
        self.dictMasks = self.oMasks.getMasks()

        ## 3D Points (camera coordinates)
        self.arrPoints = np.asarray(self.pcdRaw.points).reshape(iHeightImage, iWidthImage, 3)
        # print(self.arrPoints.shape)

        ## Create a dictionary with points of each object
        self.dictObjects = self.__createObjectsDict()

        ## Process the object point clouds
        self.dictProcessedPcds = self.__processObjects()


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
            iRawNormalRadius = 7,
            iPoissonDepth=9,
            iDensityThreshold=0.5,
            iTaubinInter=100,
            iPoints = 700,
            bVisualize=False
        )

        return pcd_reconstructed

    @staticmethod
    def __surfaceReconstruction(pointcloud: o3d.geometry.PointCloud, iRawNormalRadius: int, iPoissonDepth: int,
                                iDensityThreshold: float, iTaubinInter: int,
                                iPoints: int, bVisualize: bool) -> o3d.geometry.PointCloud:
        """
        :param pointcloud: Input point cloud for surface reconstruction.
        :param iRawNormalRadius: Radius used for initial normal estimation.
        :param iPoissonDepth: Depth parameter for the Poisson surface reconstruction algorithm.
        :param iDensityThreshold: Threshold for removing vertices based on density during reconstruction.
        :param iTaubinInter: Number of iterations for Taubin smoothing filter.
        :param iPoints: Number of points to sample after reconstruction.
        :param bVisualize: Flag to visualize intermediate steps during reconstruction.
        :return: Reconstructed point cloud after surface reconstruction and smoothing.
        """

        ## 1. First normal estimation
        # Orient normals to camera (upwards)
        pointcloud.estimate_normals()
        pointcloud.orient_normals_to_align_with_direction([0, 0, -1])

        # Recalculate normals with search parameters
        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=iRawNormalRadius)
        pointcloud.estimate_normals(oNormalSearchParam)

        if bVisualize:
            display_point_clouds(arrPointClouds=[pointcloud],
                                 sWindowTitle="Estimated normals before reconstruction",
                                 bShowOrigin=True,
                                 iOriginSize=100
            )

        ## 2. Perform poisson surface reconstruction
        mshSurfRec, arrDensities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd=pointcloud,
            depth=iPoissonDepth
        )

        arrDensities = np.asarray(arrDensities)

        if bVisualize:
            visualizeDensities(arrDensities, mshSurfRec)

        ## 3. Remove points with the lowest density
        arrVerticesToRemove = arrDensities < np.quantile(arrDensities, iDensityThreshold)
        mshSurfRec.remove_vertices_by_mask(arrVerticesToRemove)

        ## 4. Smoothing with taubin filter
        mshSurfRecSmooth = mshSurfRec.filter_smooth_taubin(number_of_iterations=iTaubinInter)
        mshSurfRecSmooth.compute_vertex_normals()

        pointcloud_reconstructed = mshSurfRecSmooth.sample_points_poisson_disk(number_of_points=iPoints)

        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=7)
        pointcloud_reconstructed.estimate_normals(oNormalSearchParam)

        return pointcloud_reconstructed



if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(Path("2025-02-20_19-46-58.ply"))

    oScene = Scene(raw_pcd=pcd,
                   iWidthImage=1920,
                   iHeightImage=1080,
                   iVoxelSize=5,
                   iOutlierNeighbours=5,
                   iStd=1)

    oScene.oMasks.debugSegmentation()

    oScene.displayObjectPoints()
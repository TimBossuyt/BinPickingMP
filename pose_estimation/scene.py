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

class Scene:
    """
    Manages scene pointcloud for pose estimation
    """

    def __init__(self, pcd, iWidthImage, iHeightImage, iVoxelSize, iOutlierNeighbours, iStd):
        ## Save settings
        self.pcdRaw = pcd
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


    def displayObjectPoints(self):
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

    def __createObjectsDict(self):
        dictObjects = {}

        ## apply mask for each detected object to the dictionary with the XYZ points as an array to corresponding key
        for _id, mask in self.dictMasks.items():
            _id = int(_id)
            dictObjects[_id] = self.arrPoints[mask==1]
            ## Remove all points with z = 0
            dictObjects[_id] = dictObjects[_id][dictObjects[_id][:,2]!=0]

        return dictObjects

    def __processObjects(self):
        pcds = {}

        for _id, points in self.dictObjects.items():
            pcds[_id] = self.__processPoints(points)

        return pcds

    def __processPoints(self, points):
        ## Returns processed pointcloud with normals

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
    def __surfaceReconstruction(pointcloud, iRawNormalRadius, iPoissonDepth,
                                iDensityThreshold, iTaubinInter, iPoints, bVisualize):

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

    oScene = Scene(pcd=pcd,
                   iWidthImage=1920,
                   iHeightImage=1080,
                   iVoxelSize=5,
                   iOutlierNeighbours=5,
                   iStd=1)

    oScene.oMasks.debugSegmentation()

    oScene.displayObjectPoints()
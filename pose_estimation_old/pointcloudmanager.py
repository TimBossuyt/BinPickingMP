import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from .utils import *

class PointCloudManager:
    """
    Manages loading, preprocessing, clustering, and reconstructing point clouds.
    """
    def __init__(self):
        self.oScene = None
        self.oModel = None

        ## Define pointcloud objects to allow easy access
        self.pcdScene = None
        self.pcdModel = None


    def loadModel(self, oModel):
        self.oModel = oModel

        self.pcdModel = self.oModel.pcdModel


    def loadScene(self, oScene, iXmin, iXmax, iYmin, iYmax, iPlaneDistance, iClusterEpsilon,
                  iClusterMinPointsDensity, iClusterPointsValidObject,
                  iReconNormalRadius, iReconPoissonDepth, iReconDensityThresh, iReconTaubinIter, iReconPoints):

        self.oScene = oScene

        ########## Perform the preprocessing functions ##########
        ### Select ROI scene ###
        self.oScene.transformAndCrop_(
            iXmin=iXmin,
            iXmax=iXmax,
            iYmin=iYmin,
            iYmax=iYmax
        )

        ### Remove background ###
        self.oScene.removeSceneBackgroundPlane_(
            iPlaneDistance=iPlaneDistance
        )

        ### Cluster scene ###
        self.oScene.clusterScene_(
            iEpsilon=iClusterEpsilon,
            iMinPointsDensity=iClusterMinPointsDensity,
            iPointsValidObject=iClusterPointsValidObject
        )

        ### Surface reconstruction of detected objects ###
        self.oScene.surfaceReconstructionObjects_(
            bVisualize=False,
            iRawNormalRadius=iReconNormalRadius,
            iPoissonDepth=iReconPoissonDepth,
            iDensityThreshold=iReconDensityThresh,
            iTaubinIter=iReconTaubinIter,
            iPoints=iReconPoints
        )

        self.pcdScene = self.oScene.pcdScene

    def getDetectedObjects(self):
        return self.oScene.arrPcdObjectsReconstructed


    #### Visualization scripts
    def displayScene(self):
        display_point_clouds([self.oScene.pcdScene], "Scene", False, True, 70)

    def displaySceneNoBackground(self):
        """Displays reconstructed objects."""
        display_point_clouds([self.oScene.pcdSceneNoBackground], "Scene - no background", False, True, 70)

    def displaySceneClusters(self):
        """Displays the clustered objects with random colors."""
        arrColoredPcds = []
        for pcd in self.oScene.arrPcdObjectsRaw:
            ## Generate random color
            oRandomColor = np.random.rand(3)

            ## Assign color to all points in pointcloud
            pcd.colors = o3d.utility.Vector3dVector(np.tile(oRandomColor, (np.asarray(pcd.points).shape[0], 1)))

            arrColoredPcds.append(pcd)

        display_point_clouds(arrColoredPcds, "Clusters", False)

    def displaySceneReconstructedObjects(self):
        """Displays reconstructed objects."""
        display_point_clouds(self.oScene.arrPcdObjectsReconstructed, "Reconstructed objects", True, True, 100)

    def displayModel(self):
        display_point_clouds([self.oModel.pcdModel], "Model", False, True, 20)











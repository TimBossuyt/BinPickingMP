import os

from pose_estimation import PointCloudManager, display_point_clouds
from pose_estimation import PoseEstimatorFPFH
import numpy as np

sModelPath = os.path.join("../Input", "T-stuk-half.stl")
sScenePath = os.path.join(
    "../PointCloudImages/PointClouds_2024-12-09_15-31-59/2024-12-09_15-32-10/PointCloud_2024-12-09_15-32-10.ply")

sExtrinsicsPath = "../CalibrationData/arrTransMat.npy"

oPointCloudManager = PointCloudManager()

########## Loading and displaying model ##########
oPointCloudManager.loadModelMesh(
    sModelPath=sModelPath,
    iPoints=2000
)

oPointCloudManager.displayModel()

## Loading and displaying scene

arrTransMat = np.load(sExtrinsicsPath)

oPointCloudManager.loadScenePointCloud(
    sScenePath=sScenePath,
    iVoxelSize=3,
    iOutlierNeighbours=20,
    iStd=1,
    iROIX=320,
    iROIY=250,
    arrWCTransform=arrTransMat,
)

oPointCloudManager.displayScene()

########## Processing the scene ##########
oPointCloudManager.removeSceneBackgroundPlane(
    iPlaneDistance=25
)

oPointCloudManager.displayNoBackgroundScene()

oPointCloudManager.clusterScene(
    iEpsilon=10,
    iMinPointsDensity=20,
    iPointsValidObject=300
)

oPointCloudManager.displayClusters()

oPointCloudManager.surfaceReconstructionObjects(
    bVisualize=False,
    iRawNormalRadius=5,
    iPoissonDepth=9,
    iDensityThreshold=0.2,
    iTaubinIter=100,
    iPoints=700
)

oPointCloudManager.displayReconstructedObjects()

### Matching
arrPcdReconstructedObjects = oPointCloudManager.arrPcdObjectsReconstructed

for pcdReconstructed in arrPcdReconstructedObjects:
    oPoseEstimator = PoseEstimatorFPFH(oPointCloudManager.pcdModel, pcdReconstructed, 3)
    oPoseEstimator.calculate_features(20, 100)

    oPoseEstimator.match(2, 1)

    trans_mat = oPoseEstimator.get_transformation_mat()

    pcdModelTransformed = oPointCloudManager.pcdModel.transform(trans_mat)

    display_point_clouds([pcdModelTransformed, pcdReconstructed, oPointCloudManager.pcdScene],
                         f"Result",
                         False, True, 100)


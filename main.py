import os
from PointCloudManager import PointCloudManager, display_point_clouds
from PoseEstimator import PoseEstimator

sModelPath = os.path.join("Input", "T-stuk-filled.stl")
sScenePath = os.path.join("PointCloudImages/PointClouds_2024-12-07_21-22-13/2024-12-07_21-22-29/PointCloud_2024-12-07_21-22-29.ply")

oPointCloudManager = PointCloudManager()

########## Loading and displaying model ##########
oPointCloudManager.loadModelMesh(
    sModelPath=sModelPath,
    iPoints=2000
)

oPointCloudManager.displayModel()

## Loading and displaying scene
oPointCloudManager.loadScenePointCloud(
    sScenePath=sScenePath,
    iVoxelSize=3,
    iOutlierNeighbours=20,
    iStd=1,
    iROIX=320,
    iROIY=250
)

oPointCloudManager.displayScene()

########## Processing the scene ##########
oPointCloudManager.removeSceneBackgroundPlane(
    iPlaneDistance=25
)

oPointCloudManager.displayNoBackgroundScene()

oPointCloudManager.clusterScene(
    iEpsilon=10,
    iMinPointsDensity=10,
    iPointsValidObject=100
)

oPointCloudManager.displayClusters()

oPointCloudManager.surfaceReconstructionObjects(
    bVisualize=True,
    iRawNormalRadius=20,
    iPoissonDepth=9,
    iDensityThreshold=0.2,
    iTaubinIter=100,
    iPoints=2000
)

oPointCloudManager.displayReconstructedObjects()

### Matching
arrPcdReconstructedObjects = oPointCloudManager.arrPcdObjectsReconstructed

print(len(arrPcdReconstructedObjects))


for pcdReconstructed in arrPcdReconstructedObjects:
    oPoseEstimator = PoseEstimator(oPointCloudManager.pcdModel, pcdReconstructed, 3)
    oPoseEstimator.calculate_features(20, 100)

    oPoseEstimator.match(3)

    trans_mat = oPoseEstimator.get_transformation_mat()

    pcdModelTransformed = oPointCloudManager.pcdModel.transform(trans_mat)

    display_point_clouds([pcdModelTransformed, pcdReconstructed, oPointCloudManager.pcdScene],
                         f"Result",
                         False, True, 100)
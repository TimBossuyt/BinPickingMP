from pose_estimation import Scene
import numpy as np

sScenePath = "../PointCloudImages/PointClouds_2024-12-09_15-31-59/2024-12-09_15-32-10/PointCloud_2024-12-09_15-32-10.ply"

sExtrinsicsPath = "../CalibrationData/arrTransMat.npy"

arrTransMat = np.load(sExtrinsicsPath)

oScene = Scene(
    sScenePath=sScenePath,
    arrWCTransform=arrTransMat,
    iVoxelSize=3,
    iOutlierNeighbours=20,
    iStd=1
)

# ### Scene processing
# oScene.removeSceneBackgroundPlane(
#     iPlaneDistance=25
# )
#
# oScene.__clusterScene(
#     iEpsilon=10,
#     iMinPointsDensity=20,
#     iPointsValidObject=300
# )
#
# oScene.surfaceReconstructionObjects(
#     bVisualize=False,
#     iRawNormalRadius=5,
#     iPoissonDepth=9,
#     iDensityThreshold=0.2,
#     iTaubinIter=100,
#     iPoints=700
# )
#
#
# oScene.displayScene()
# oScene.displayNoBackgroundScene()
# oScene.displayClusters()
# oScene.displayReconstructedObjects()


from pose_estimation_old import Model, Scene, PointCloudManager, PoseEstimatorFPFH
from utils import display_point_clouds
import numpy as np

## Parameters
sModelPath = "../Input/T-stuk-filled.stl"
sScenePath = "../PointCloudImages/PointClouds_2024-12-09_15-31-59/2024-12-09_15-32-10/PointCloud_2024-12-09_15-32-10.ply"
sExtrinsicsPath = "../CalibrationData/arrTransMat.npy"

## Loading extrinsic camera parameters
arrTransMat = np.load(sExtrinsicsPath)

oModel = Model(sModelPath, 2000, 10)

oScene = Scene(
    sScenePath=sScenePath,
    arrWCTransform=arrTransMat,
    iVoxelSize=3,
    iOutlierNeighbours=20,
    iStd=1,
)

oPointCloudManager = PointCloudManager()

oPointCloudManager.loadModel(oModel)

oPointCloudManager.loadScene(
    oScene=oScene,
    iXmin=0,
    iXmax=320,
    iYmin=0,
    iYmax=250,
    iPlaneDistance=25,
    iClusterEpsilon=10,
    iClusterMinPointsDensity=20,
    iClusterPointsValidObject=300,
    iReconNormalRadius=5,
    iReconPoissonDepth=9,
    iReconDensityThresh=0.2,
    iReconTaubinIter=100,
    iReconPoints=700
)

oPointCloudManager.displayModel()
# oPointCloudManager.displaySceneReconstructedObjects()

arrPcdDetectedObjects = oPointCloudManager.getDetectedObjects()

for pcdReconstructed in arrPcdDetectedObjects:
    oPoseEstimator = PoseEstimatorFPFH(oPointCloudManager.pcdModel, pcdReconstructed, 3)
    oPoseEstimator.calculate_features(20, 100)

    oPoseEstimator.match(2, 1)

    trans_mat = oPoseEstimator.get_transformation_mat()

    pcdModelTransformed = oPointCloudManager.pcdModel.transform(trans_mat)

    display_point_clouds([pcdModelTransformed, pcdReconstructed, oPointCloudManager.pcdScene],
                         f"Result",
                         False, True, 100)



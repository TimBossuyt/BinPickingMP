import open3d as o3d
from pathlib import Path
from pose_estimation import *


## Load settings
sm = SettingsManager("./test_input/default_settings.json")

## Load model
oModel = Model(
    sModelPath="./test_input/T-stuk-filled.stl",
    settingsManager=sm,
    picking_pose=None,
)

## Load scene
pcd = o3d.io.read_point_cloud(Path("./test_input/2025-02-20_19-46-58.ply"))

oScene = Scene(
    raw_pcd=pcd,
    settingsmanager=sm
)

## Create pose estimator object
oPoseEstimator = PoseEstimatorFPFH(
    settingsManager=sm,
    model=oModel
)

transforms = oPoseEstimator.findObjectTransforms(oScene)

oTransformVisualizer = TransformVisualizer(oModel, oScene, transforms)
oTransformVisualizer.displayFoundObjects()
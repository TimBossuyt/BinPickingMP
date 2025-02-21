from model import Model
from scene import Scene
from settings import SettingsManager
from utils import display_point_clouds
from pose_estimator import PoseEstimatorFPFH
import open3d as o3d
from pathlib import Path


class TransformVisualizer:
    def __init__(self, model: Model, scene: Scene, transformations: dict):
        self.oModel = model
        self.oScene = scene

        self.dictTransforms = transformations

    def displayFoundObjects(self):
        pcds = [self.oScene.pcdRaw]

        for _id, transform in self.dictTransforms.items():
            pcdModel = self.oModel.pcdModel

            pcdModelCopy = o3d.geometry.PointCloud()
            pcdModelCopy.points = self.oModel.pcdModel.points

            pcdObjectTransformed = pcdModelCopy.transform(transform)
            pcds.append(pcdObjectTransformed)


        display_point_clouds(pcds, "Found objects", False, True, 100)


if __name__ == "__main__":
    ## Load settings
    sm = SettingsManager("default_settings.json")

    ## Load model
    oModel = Model(
        sModelPath="test_input/T-stuk-filled.stl",
        settingsManager=sm,
        picking_pose=None,
    )

    ## Load scene
    pcd = o3d.io.read_point_cloud(Path("test_input/2025-02-20_19-46-58.ply"))

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


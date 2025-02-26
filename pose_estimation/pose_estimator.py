from .scene import Scene
from .model import Model
from .settings import SettingsManager
import open3d as o3d
import numpy as np
from pathlib import Path


class PoseEstimatorFPFH:
    def __init__(self, settingsManager: SettingsManager , model: Model):
        self.oModel = model

        ## Settings
        self.oSm = settingsManager
        self.__loadSettings()

        ## Calculate model features
        self.__calculateModelFeatures()


    def findObjectTransforms(self, scene: Scene) -> dict:
        results = {}

        for _id, pcdObject in scene.dictProcessedPcds.items():
            results[_id] = self.__calculateTransform(pcdObject)

        return results

    def __calculateTransform(self, pcdObject: o3d.geometry.PointCloud) -> np.ndarray:
        ## 1. Voxel down object pointcloud to same density as object model
        pcdObjectDown = pcdObject.voxel_down_sample(voxel_size=self.iVoxelSize)

        ## 2. Calculate the FPFH features
        oObjectFPFH = o3d.pipelines.registration.compute_fpfh_feature(
            input=pcdObjectDown,
            search_param=self.oFeatureParams
        )

        ## 3. Find initial transformation using RANSAC
        oInitialMatch = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=self.pcdModelDown,
            target=pcdObjectDown,
            source_feature=self.oModelFPFH,
            target_feature=oObjectFPFH,
            mutual_filter=False,
            max_correspondence_distance=self.distanceFactor*self.iVoxelSize,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.distanceFactor*self.iVoxelSize),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        ## 4. Perform ICP for finer results
        oIcpResult = o3d.pipelines.registration.registration_icp(
            source=self.pcdModelDown,
            target=pcdObjectDown,
            max_correspondence_distance=self.iVoxelSize*self.icpDistanceFactor,
            init=oInitialMatch.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )


        # # Visualize (DEBUGGING ONLY)
        # pcdModelTransformed = self.pcdModelDown.transform(np.asarray(oIcpResult.transformation))
        #
        # display_point_clouds(
        #     [
        #         pcdObjectDown,
        #         pcdModelTransformed,
        #     ],
        #     "Object and model after ICP",
        #     False, True, 100)

        return np.asarray(oIcpResult.transformation)

    def __loadSettings(self):
        ## General
        self.iVoxelSize = self.oSm.get("PoseEstimation.General.VoxelSize")

        ## Feature params
        self.featureFactor = self.oSm.get("PoseEstimation.FeatureParams.FeatureFactor")
        self.featureMaxNeighbours = self.oSm.get("PoseEstimation.FeatureParams.MaxNeighbours")

        ## Save search parameters
        self.oFeatureParams = o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.featureFactor*self.iVoxelSize,
            max_nn=self.featureMaxNeighbours
        )

        ## Matching parameters
        self.distanceFactor = self.oSm.get("PoseEstimation.Matching.DistanceFactor")
        self.icpDistanceFactor = self.oSm.get("PoseEstimation.Matching.IcpDistanceFactor")


    def __calculateModelFeatures(self):
        ## Voxel down (make sure scene model points are same density
        self.pcdModelDown = self.oModel.pcdModel.voxel_down_sample(voxel_size=self.iVoxelSize)

        ## Calculating model features
        self.oModelFPFH = o3d.pipelines.registration.compute_fpfh_feature(
            input=self.pcdModelDown,
            search_param=self.oFeatureParams
        )

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
        raw_pcd = pcd,
        settingsmanager=sm
    )

    ## Create pose estimator object
    oPoseEstimator = PoseEstimatorFPFH(
        settingsManager=sm,
        model=oModel
    )

    print(oPoseEstimator.findObjectTransforms(oScene))




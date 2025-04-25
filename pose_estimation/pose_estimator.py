from IPython.core.display_functions import display

from pose_estimation.utils import display_point_clouds
from .scene import Scene
from .model import Model
from .settings import SettingsManager
import open3d as o3d
import numpy as np
from pathlib import Path
import time
import logging

logger = logging.getLogger("Pose Estimator")

class PoseEstimatorFPFH:
    def __init__(self, settingsManager: SettingsManager , model: Model):
        ## Settings
        self.oSm = settingsManager
        self.__loadSettings()

        self.oModel = model

        ## Calculate model features (only 1 time)
        self.__calculateModelFeatures()

    def reload_settings(self):
        self.__loadSettings()
        logger.info("Reloaded settings")

    def findObjectTransforms(self, scene: Scene) -> dict[int, tuple[np.ndarray, float, float]]:
        """
        Main method for finding object transforms given a scene object
        :param scene:
        :return:
        """

        results = {}
        tStartTotal = time.time()

        for _id, pcdObject in scene.dictProcessedPcds.items():
            tStartObject = time.time()
            results[_id] = self.__calculateTransform(pcdObject, self.MatchingTimeOut)
            tEndObject = time.time()

            logger.debug(f"Finding object {_id} took {(tEndObject-tStartObject)*1000:.2f} ms")

        tEndTotal = time.time()

        logger.info(f"Finding object transformations took {(tEndTotal-tStartTotal)*1000:.2f} ms")

        return results

    def __calculateTransform(self, pcdObject: o3d.geometry.PointCloud, timeout: float) -> (np.ndarray, float, float):
        ## REMOVE, DEBUGGING ONLY
        # self.__calculateModelFeatures()

        ## 1. Voxel down object pointcloud to same density as object model
        pcdObjectDown = pcdObject.voxel_down_sample(voxel_size=self.iVoxelSize)

        if self.bVisualize:
            display_point_clouds([pcdObjectDown, self.pcdModelDown], "Model and object used for pose estimation",
                                 True, True, 100)

        ## 2. Calculate the FPFH features
        oObjectFPFH = o3d.pipelines.registration.compute_fpfh_feature(
            input=pcdObjectDown,
            search_param=self.oFeatureParams
        )
        if self.bVisualize:
            display_point_clouds([pcdObjectDown, self.pcdModelDown],
                                 "Pointclouds src and dst", True, True, 100)

        # 3. Find initial transformation using RANSAC
        oInitialMatch = None
        fitness = -1
        start_time = time.time()

        while fitness < self.FitnessThreshold:
            elapsed_time = (time.time() - start_time) * 1000
            if elapsed_time > timeout:
                logger.info("RANSAC timeout reached, returning default transformation")
                return np.eye(4), 0.0, 0.0  # Return identity matrix and zero fitness/RMSE


            oInitialMatch = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source=self.pcdModelDown,
                target=pcdObjectDown,
                source_feature=self.oModelFPFH,
                target_feature=oObjectFPFH,
                mutual_filter= self.bMutualFilter,
                max_correspondence_distance=self.distanceFactor*self.iVoxelSize,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=self.iRansacIterations,
                checkers=[
                    # o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    #     self.distanceFactor*self.iVoxelSize),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.CheckerLengthThreshold ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(self.NormalAngleThresh)

                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999)
            )
            fitness = oInitialMatch.fitness

        if self.iMaxIcpIterations == 0:
            logger.debug("ICP Was disabled")
            return np.asarray(oInitialMatch.transformation), oInitialMatch.fitness, oInitialMatch.inlier_rmse


        ## 4. Perform ICP for finer results
        oIcpResult = o3d.pipelines.registration.registration_icp(
            source=self.pcdModelDown,
            target=pcdObjectDown,
            max_correspondence_distance=self.iVoxelSize*self.icpDistanceFactor,
            init=oInitialMatch.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.iMaxIcpIterations)
        )

        return np.asarray(oIcpResult.transformation), oIcpResult.fitness, oIcpResult.inlier_rmse

    def __loadSettings(self):
        ## General
        self.iVoxelSize = self.oSm.get("PoseEstimation.General.VoxelSize")
        value = self.oSm.get("PoseEstimation.General.bVisualize")
        self.bVisualize = (value == 1)

        ## Feature params
        self.featureFactor = self.oSm.get("PoseEstimation.FeatureParams.FeatureFactor")
        self.featureMaxNeighbours = self.oSm.get("PoseEstimation.FeatureParams.MaxNeighbours")

        ## Save search parameters
        self.oFeatureParams = o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.featureFactor*self.iVoxelSize,
            max_nn=self.featureMaxNeighbours
        )

        ## Matching parameters
        self.iRansacIterations = self.oSm.get("PoseEstimation.Matching.RansacIterations")
        self.distanceFactor = self.oSm.get("PoseEstimation.Matching.DistanceFactor")
        self.icpDistanceFactor = self.oSm.get("PoseEstimation.Matching.IcpDistanceFactor")
        value = self.oSm.get("PoseEstimation.Matching.MutualFilter")
        self.bMutualFilter = (value == 1)

        self.CheckerLengthThreshold = self.oSm.get("PoseEstimation.Matching.CheckerEdgeLengthThreshold")
        self.NormalAngleThresh = self.oSm.get("PoseEstimation.Matching.NormalAngleThreshold")

        self.iMaxIcpIterations = self.oSm.get("PoseEstimation.Matching.MaxIcpIterations")
        self.FitnessThreshold = self.oSm.get("PoseEstimation.Matching.InitFitnessThresh")
        self.MatchingTimeOut = self.oSm.get("PoseEstimation.Matching.TimeOut")


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




import copy
from pose_estimation.utils import display_point_clouds
from .scene import Scene
from .model import Model
from .settings import SettingsManager
import open3d as o3d
import numpy as np
import time
import logging
from shapely.geometry import Polygon

logger = logging.getLogger("Pose Estimator")

class PoseEstimatorFPFH:
    def __init__(self, settingsManager: SettingsManager , model: Model):
        ## Settings
        self.oSm = settingsManager
        self._loadSettings()

        self.oModel = model

        ## Calculate model features (only 1 time)
        self._calculateModelFeatures()

    def reload_settings(self):
        self._loadSettings()
        logger.info("Reloaded settings")
        self._calculateModelFeatures()

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

    def __calculateTransform(self, pcdObject: o3d.geometry.PointCloud, timeout: float) -> (np.ndarray, float, float, float):
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

        # 3. Find initial transformation using RANSAC
        oInitialMatch = None
        fitness = -1
        iou_score = -1
        start_time = time.time()

        while (iou_score < self.IoUThreshold) or (fitness < self.FitnessThreshold):
            elapsed_time = (time.time() - start_time) * 1000
            if elapsed_time > timeout:
                logger.info("RANSAC timeout reached, returning default transformation")
                return np.eye(4), 0.0, 0.0, 0.0  # Return identity matrix and zero fitness/RMSE

            oInitialMatch = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source=self.pcdModelDown,
                target=pcdObjectDown,
                source_feature=self.oModelFPFH,
                target_feature=oObjectFPFH,
                mutual_filter= self.bMutualFilter,
                max_correspondence_distance=self.distanceFactor*self.iVoxelSize,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=self.iRansacCorrespondences,
                checkers=[
                    # o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    #     self.distanceFactor*self.iVoxelSize),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.CheckerLengthThreshold ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(self.NormalAngleThresh)

                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(self.iRansacMaxIterations, 0.9999)
            )
            fitness = oInitialMatch.fitness

            ## 5. Calculate the IoU metric from the bounding boxes in the x, y plane
            pcd_model = copy.deepcopy(self.pcdModelDown)
            iou_score = self._compute_iou(pcd_model, pcdObjectDown, np.asarray(oInitialMatch.transformation))

            print(f"Fitness: {fitness:.4f} | IoU: {iou_score:.4f} | Time: {elapsed_time:.2f} ms")

        if self.iMaxIcpIterations == 0:
            logger.debug("ICP Was disabled")
            return np.asarray(oInitialMatch.transformation), oInitialMatch.fitness, oInitialMatch.inlier_rmse, 0


        ## 4. Perform ICP for finer results
        oIcpResult = o3d.pipelines.registration.registration_icp(
            source=self.pcdModelDown,
            target=pcdObjectDown,
            max_correspondence_distance=self.iVoxelSize*self.icpDistanceFactor,
            init=oInitialMatch.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.iMaxIcpIterations)
        )

        ## 5. Calculate the IoU metric from the bounding boxes in the x, y plane
        pcd_model = copy.deepcopy(self.pcdModelDown)
        iou_score = self._compute_iou(pcd_model, pcdObjectDown, np.asarray(oIcpResult.transformation))

        return np.asarray(oIcpResult.transformation), oIcpResult.fitness, oIcpResult.inlier_rmse, iou_score

    @staticmethod
    def _compute_iou(pcd_model: o3d.geometry.PointCloud,
                     pcd_scene: o3d.geometry.PointCloud,
                     transformation: np.ndarray) -> float:
        bbox_1 = pcd_scene.get_oriented_bounding_box()
        bbox_1_points = np.asarray(bbox_1.get_box_points())

        pcd_model = pcd_model.transform(transformation)
        bbox_2 = pcd_model.get_oriented_bounding_box()
        bbox_2_points = np.asarray(bbox_2.get_box_points())

        # ## Debugging
        # display_point_clouds([pcd_model, pcd_scene], "Model and object used for IoU calculation"
        #                      , False, True, 100)

        bbox_1.color = [1, 0, 0]
        bbox_2.color = [0, 0, 1]
        # o3d.visualization.draw_geometries([bbox_1, bbox_2])

        xy_1 = [(float(bbox_1_points[i][0]), float(bbox_1_points[i][1])) for i in [0, 1, 7, 2]]
        xy_2 = [(float(bbox_2_points[i][0]), float(bbox_2_points[i][1])) for i in [0, 1, 7, 2]]

        # print(xy_1)
        # print(xy_2)

        poly1 = Polygon(xy_1)
        poly2 = Polygon(xy_2)

        # Compute intersection and union areas
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        # Calculate IoU
        iou = intersection / union
        # print(iou)

        return iou

    def _loadSettings(self):
        ## General
        self.iVoxelSize = self.oSm.get("PoseEstimation.General.VoxelSize")
        value = self.oSm.get("PoseEstimation.General.bVisualize")
        self.bVisualize = (value == 1)

        ## Feature params
        self.featureFactor = self.oSm.get("PoseEstimation.FeatureParams.FeatureFactor")

        self.oFeatureParams = o3d.geometry.KDTreeSearchParamRadius(
            radius=self.featureFactor*self.iVoxelSize,
        )

        ## Matching parameters
        self.iRansacCorrespondences = self.oSm.get("PoseEstimation.Matching.RansacCorrespondences")
        self.iRansacMaxIterations = self.oSm.get("PoseEstimation.Matching.RansacIterations")
        self.distanceFactor = self.oSm.get("PoseEstimation.Matching.DistanceFactor")
        self.icpDistanceFactor = self.oSm.get("PoseEstimation.Matching.IcpDistanceFactor")

        value = self.oSm.get("PoseEstimation.Matching.MutualFilter")
        self.bMutualFilter = (value == 1)

        self.CheckerLengthThreshold = self.oSm.get("PoseEstimation.Matching.CheckerEdgeLengthThreshold")
        self.NormalAngleThresh = self.oSm.get("PoseEstimation.Matching.NormalAngleThreshold")

        self.iMaxIcpIterations = self.oSm.get("PoseEstimation.Matching.MaxIcpIterations")
        self.FitnessThreshold = self.oSm.get("PoseEstimation.Matching.InitFitnessThresh")
        self.IoUThreshold = self.oSm.get("PoseEstimation.Matching.InitIoUThresh")
        self.MatchingTimeOut = self.oSm.get("PoseEstimation.Matching.TimeOut")


    def _calculateModelFeatures(self):
        ## Voxel down (make sure scene model points are same density
        self.pcdModelDown = self.oModel.pcdModel.voxel_down_sample(voxel_size=self.iVoxelSize)

        ## Calculating model features
        self.oModelFPFH = o3d.pipelines.registration.compute_fpfh_feature(
            input=self.pcdModelDown,
            search_param=self.oFeatureParams
        )



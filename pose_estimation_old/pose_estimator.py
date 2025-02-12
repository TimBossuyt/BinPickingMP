import numpy as np
import open3d as o3d
from .utils import centroid_to_centroid_transformation

## TODO: Make it work for an array of scene-objects

class PoseEstimatorFPFH:
    def __init__(self, pcdModel, pcdScene, iVoxelSize):
        self.pcdModel = pcdModel
        self.pcdScene = pcdScene
        self.iVoxelSize = iVoxelSize

        ## Initial transformation
        self.pcdModel = pcdModel.transform(centroid_to_centroid_transformation(self.pcdModel, self.pcdScene))

        ## Downsample both pointclouds to get the same density
        self.pcdModel = self.pcdModel.voxel_down_sample(self.iVoxelSize)
        self.pcdScene = self.pcdScene.voxel_down_sample(self.iVoxelSize)

        self.oModelFPFH = None
        self.oSceneFPFH = None

        self.oMatchingResults = None

        self.oTransformation = None


    def calculate_features(self, iFeatureFactor, iMaxNN):
        iRadiusFeature = self.iVoxelSize * iFeatureFactor

        ## Calculate model features (FPFH)
        self.oModelFPFH = o3d.pipelines.registration.compute_fpfh_feature(self.pcdModel,
                                                                     o3d.geometry.KDTreeSearchParamHybrid(
                                                                         radius=iRadiusFeature, max_nn=iMaxNN))

        ## Calculate scene/object features (FPFH)
        self.oSceneFPFH = o3d.pipelines.registration.compute_fpfh_feature(self.pcdScene,
                                                                     o3d.geometry.KDTreeSearchParamHybrid(
                                                                         radius=iRadiusFeature, max_nn=iMaxNN))

    def match(self, iDistanceFactor, iICPDistanceFactor=None):
        iDistanceThreshold = self.iVoxelSize * iDistanceFactor

        self.oMatchingResults = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=self.pcdModel,
            target=self.pcdScene,
            source_feature=self.oModelFPFH,
            target_feature=self.oSceneFPFH,
            mutual_filter=False,
            max_correspondence_distance=iDistanceThreshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    iDistanceThreshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        self.oTransformation = self.oMatchingResults.transformation

        if iICPDistanceFactor is not None:
            iICPThreshold = self.iVoxelSize * iICPDistanceFactor

            ## Function requires an initial transformation!!
            icp_result = o3d.pipelines.registration.registration_icp(
                source=self.pcdModel,
                target=self.pcdScene,
                max_correspondence_distance=iICPThreshold,
                init=self.oTransformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )

            # Update the transformation with the refined ICP result
            self.oTransformation = icp_result.transformation

    def get_transformation_mat(self):
        return np.array(self.oTransformation, copy=True)
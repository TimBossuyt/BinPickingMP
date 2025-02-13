from .roi_aruco import ExtrinsicTransformation
import cv2
import logging

logger = logging.getLogger("Pose-Estimator")

class PoseEstimator:
    def __init__(self):
        ## Setup Extrinsic transformation object
        self.oExtrinsicTransformation = ExtrinsicTransformation(
            oArucoDict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
            iHeightImage=1080,
            iWidthImage=1920
        )

    def DetectMarkers(self, cvImage):
        logger.info("Detecting AruCo markers")
        return self.oExtrinsicTransformation.showMarkers(cvImage)


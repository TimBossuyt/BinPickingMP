import numpy as np
import cv2
import logging

logger = logging.getLogger("Calibration")

def drawDetectedCorners(image, charuco_corners, charuco_ids):
    """
    Creates annotated image based on the detected corners with ids
    """

    img_result = image.copy()
    cv2.aruco.drawDetectedCornersCharuco(
        image=img_result,
        charucoIds=charuco_ids,
        charucoCorners=charuco_corners
    )

    ## Highlight the detected corners
    for corner in charuco_corners:
        cv2.circle(img_result, tuple(corner[0].astype(int)), 5, (0, 255, 0), -1)

    return img_result


class BoardDetector:
    def __init__(self, iSquareLength, iMarkerLength, size: tuple[int, int]):
        self.oArucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

        ## Define the charuco board
        self.oChArUcoBoard = cv2.aruco.CharucoBoard(
            size=size,
            markerLength=iMarkerLength,
            squareLength=iSquareLength,
            dictionary=self.oArucoDict
        )

        ## Initialize the detector
        self.oDetector = cv2.aruco.CharucoDetector(
            self.oChArUcoBoard
        )

    def detectBoard(self, image):
        ## Find corners and corresponding ids from the board
        charuco_corners, charuco_ids, _, _ = self.oDetector.detectBoard(image=image)

        if charuco_corners is None or charuco_ids is None:
            logger.error("No board was found")
            return None, None  # No board detected

        return charuco_corners, charuco_ids


class CameraCalibrator:
    def __init__(self, arrCameraMatrix, arrDistortionCoeffs=None):
        if arrDistortionCoeffs is None:
            arrDistortionCoeffs = [0, 0, 0, 0, 0]
        logger.debug("CameraCalibrator initializing")
        self.arrCameraMatrix = np.asarray(arrCameraMatrix)
        self.arrDistortionCoeffs = np.asarray(arrDistortionCoeffs)

        ## TODO: Remove hardcoding
        self.oBoardDetector = BoardDetector(
            iSquareLength=37.5,
            iMarkerLength=25,
            size=(7, 5),
        )

        ## Checks if the calibration has completed
        self.bCalibrated = False

        ## rvec and tvec save the transformation to go from board space to camera space
        self.rvec_bc = None
        self.tvec_bc = None

        ## Image with board used for calibration
        self.calibrationImage = None

        ## Save detected charuco corners and ids (in image coordinates [pixels])
        self.arrChArUcoCorners = None
        self.arrChArUcoIds = None

        ## Detected corners in Board coordinates (3D [mm])
        self.arrBoardCornerPoints = None #3D

        ## Detected corners in image coordinates (2D [pixels])
        self.arrImgCornerPoints = None #2D

        ## Dictionary of point coordinates with corresponding id as key
        self.dictWorldPoints = {} # In world coordinates
        self.dictCameraPoints = {} # In camera coordinates

        ## Array of points in each coordinate space to estimate transformation
        # Point in one array should correspond to the point with the same index in the other array
        self.arrWorldPoints = []
        self.arrCamPoints = []

    def runCalibration(self, image, dictWorldPoints):
        """
        Run full calibration from given image and dictionary of world points
        """

        ## Save as attributes
        self.calibrationImage = image
        self.dictWorldPoints = dictWorldPoints

        ## 1. Detect board and save corners/ids
        self.arrChArUcoCorners, self.arrChArUcoIds = self.oBoardDetector.detectBoard(image)
        if self.arrChArUcoCorners is None or self.arrChArUcoIds is None:
            logger.warning("No board was found")

        ## 2. Estimate the board pose w.r.t. camera
        # Calculates transformation needed to go from board coordinates to camera coordinates
        logger.debug("Estimating board pose")
        self.__estimateBoardPose()

        ## 3. Create corner points dictionary in camera coordinates
        self.__calculateCameraDict()

        ## 4. Add depth information to the dictionaries
        self.__includeDepthInfo()

        ## 5. Create list of corresponding points
        self.__createCorrespondingPointsArray()

        ## 6. Calibrate camera-to-world transformation
        logger.debug("Calculating camera-world transformation")
        trans_mat = self.calibrateCameraWorldTransformation()
        logger.info("Found camera-world transformation")

        self.bCalibrated = True

        return trans_mat

    def showDetectedBoard(self):
        """
        Returns an image with annotated corners/ids of the charuco board
        """
        if self.bCalibrated:
            ## Call static function to annotate the image
            img = drawDetectedCorners(self.calibrationImage,
                                      self.arrChArUcoCorners,
                                      self.arrChArUcoIds)

            return img
        else:
            logger.warning("First run calibration")
            return None

    def saveCornerCameraCoordinates(self):
        """
        Save corners in camera coordinates to a .txt file for debugging
        """
        if self.bCalibrated:
            output_file = "charuco_camera_coordinates.txt"
            with open(output_file, "w") as f:
                for charuco_id, coord in self.dictCameraPoints.items():
                    coord_str = " ".join(map(str, coord))  # Convert coordinates to a space-separated string
                    f.write(f"ChArUco ID {charuco_id}: Camera Coordinates {coord_str}\n")

            logger.info(f"ChArUco camera coordinates saved to {output_file}")
        else:
            logger.warning("First run calibration")

    def __estimateBoardPose(self):
        """
        Finds the rvec and tvec needed to go from board coordinates to camera coordinates
        """

        ## Get points in charuco space (3D) and in image space (pixels)
        self.arrBoardCornerPoints, self.arrImgCornerPoints = self.oBoardDetector.oChArUcoBoard.matchImagePoints(
            detectedCorners=self.arrChArUcoCorners,
            detectedIds=self.arrChArUcoIds,
        )

        ## Find the transformation that maps points in charuco space to camera space
        _, rvec, tvec = cv2.solvePnP(
            objectPoints=self.arrBoardCornerPoints,
            imagePoints=self.arrImgCornerPoints,
            cameraMatrix=self.arrCameraMatrix,
            distCoeffs=self.arrDistortionCoeffs,
        )

        self.rvec_bc = rvec
        self.tvec_bc = tvec

        return rvec, tvec

    def calibrateCameraWorldTransformation(self):
        """
        Calculates transformation matrix to go from camera coordinates to world coordinates
        """

        _, out, inliers = cv2.estimateAffine3D(self.arrCamPoints, self.arrWorldPoints, True)

        trans_mat = np.vstack((out, [0, 0, 0, 1]))
        logger.debug(f"Calculated trans_mat: {trans_mat}")

        return trans_mat

    def __calculateCameraDict(self):
        """
        Save the detected corners as a dictionary (in camera coordinates)
        """
        charuco_3D_camera = {}
        for i, charuco_id in enumerate(self.arrChArUcoIds.flatten()):
            board_point = self.arrBoardCornerPoints[charuco_id].flatten()
            camera_point = self.__convertBoardToCamera(board_point)

            charuco_3D_camera[charuco_id] = camera_point

        ## Save
        self.dictCameraPoints = charuco_3D_camera

    def __convertBoardToCamera(self, board_point):
        """
        Transformation function to transform from board coordinates to camera coordinates
        """
        R, _ = cv2.Rodrigues(self.rvec_bc)
        camera_point = R@np.array(board_point) + self.tvec_bc.flatten()

        return camera_point

    def __includeDepthInfo(self):
        """
        Add depth information to the points to get a non-zero z-transformation
        """

        # TODO: Automate the included depth information
        # Point along z-axis with offset 50 from surface
        self.dictWorldPoints[100] = [0, 0, 50]

        # Same point in camera coordinates
        # = negative z-offset from origin in board coordinates transformed to camera coordinates
        point = [37.5, 37.5, -50]
        point_camera = self.__convertBoardToCamera(point)
        self.dictCameraPoints[100] = point_camera

    def __createCorrespondingPointsArray(self):
        """
        Create the two arrays that are used to find the transformation
        """

        ## Reset arrays
        self.arrWorldPoints = []
        self.arrCamPoints = []

        ## Add each point with id from the world points dictionary
        # + find point with corresponding id in the camera points list and add it to the array
        for (id, point) in self.dictWorldPoints.items():
            self.arrWorldPoints.append(point)
            self.arrCamPoints.append(self.dictCameraPoints[id].tolist())

        ## Reshape arrays to numpy format
        self.arrWorldPoints = np.array(self.arrWorldPoints, dtype=np.float32).reshape(-1, 3)
        self.arrCamPoints = np.array(self.arrCamPoints, dtype=np.float32).reshape(-1, 3)



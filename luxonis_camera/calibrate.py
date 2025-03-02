import numpy as np
import cv2
import logging

logger = logging.getLogger("Calibration")

def drawDetectedCorners(image, charuco_corners, charuco_ids):
    """
    :param image: The input image on which the detected ChArUco corners will be drawn.
    :param charuco_corners: The coordinates of the detected ChArUco corners.
    :param charuco_ids: The IDs associated with the detected ChArUco corners.
    :return: A copy of the input image with the detected ChArUco corners and their highlights drawn.
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
    """
    BoardDetector is a class designed to detect a ChArUco board in a given image using the OpenCV ArUco module.

    It provides functionality to configure the board with specified parameters
    and detect its corners and IDs from an input image.

    Attributes:
        oArucoDict: The predefined ArUco dictionary used for marker detection.
        oChArUcoBoard: The ChArUcoBoard object representing the board to be detected.
        oDetector: The detector object initialized witth the ChArUco board for performing board detection.

    Methods:
        __init__(iSquareLength, iMarkerLength, size):
            Initializes the BoardDetector object with the square length of the ChArUco board,
            marker length, and the dimensions of the board.

        detectBoard(image):
            Detects the ChArUco board in the provided image.
            Returns the detected corners and corresponding IDs if a board is successfully detected, otherwise returns None.
    """
    def __init__(self, iSquareLength, iMarkerLength, size: tuple[int, int]):
        """
        :param iSquareLength: The length of the squares on the ChArUco board, specified as a float or an int.
        :param iMarkerLength: The length of the markers on the ChArUco board, specified as a float or an int.
        :param size: A tuple of two integers representing the number of squares along the board's width and height.
        """

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
        """
        :param image: The input image in which the board is to be detected.
        :return: A tuple containing detected charuco corners and their corresponding ids.
                    Returns (None, None) if no tboard is detected.
        """

        ## Find corners and corresponding ids from the board
        charuco_corners, charuco_ids, _, _ = self.oDetector.detectBoard(image=image)

        if charuco_corners is None or charuco_ids is None:
            logger.error("No board was found")
            return None, None  # No board detected

        return charuco_corners, charuco_ids


class CameraCalibrator:
    """
    CameraCalibrator class is designed for performing camera calibration using a planar ChArUco board as the target.

    The class estimates the transformation matrices between various coordinate systems
    such as camera, board, and world coordinates.

    Attributes:
        arrCameraMatrix: 2D NumPy array storing the intrinsic camera matrix for the camera.
        arrDistortionCoeffs: NumPy array storing the camera's distortion coefficients. Defaults to [0, 0, 0, 0, 0].
        oBoardDetector: Instance of `BoardDetector`, responsible for detecting the ChArUco board in input images.
        bCalibrated: Boolean flag indicating whether the calibration process has been completed.
        rvec_bc: NumPy array representing the rotation vector for the transformation from board to camera coordinates.
        tvec_bc: NumPy array representing the translation vector for the transformation from board to camera coordinates.
        calibrationImage: Image used for performing the calibration.
        arrChArUcoCorners: Detected corners of the ChArUco board, in image pixel coordinates.
        arrChArUcoIds: Detected marker IDs corresponding to ChArUco corners.
        arrBoardCornerPoints: NumPy array of detected corner points in the board's local 3D coordinate system.
        arrImgCornerPoints: NumPy array of detected corner points in the image's 2D pixel coordinate system.
        dictWorldPoints: Dictionary mapping point IDs to their 3D coordinates in the world coordinate system.
        dictCameraPoints: Dictionary mapping point IDs to their 3D coordinates in the camera coordinate system.
        arrWorldPoints: List of 3D points used for finding the world-to-camera transformation.
        arrCamPoints: List of corresponding 3D camera points for finding the world-to-camera transformation.

    Methods:
        runCalibration(image, dictWorldPoints):
            Perform the entire calibration procedure using an input image and a dictionary of world points.

        showDetectedBoard():
            Generate an annotated image showing ChArUco corners and their IDs if calibration succeeded.

        saveCornerCameraCoordinates():
            Save 3D coordinates of detected camera points to a text file. (Debugging purposes)

        __estimateBoardPose():
            Estimate the pose of the board in relation to the camera by calculating the rotation and translation vectors.

        calibrateCameraWorldTransformation():
            Compute the transformation matrix needed to convert coordinates from the camera reference frame to the world reference frame.

        __calculateCameraDict():
            Store detected corner points in the camera coordinate system as a dictionary.

        __convertBoardToCamera(board_point):
            Transform a 3D point from the board coordinate system to the camera coordinate system.

        __includeDepthInfo():
            Add depth information manually to generate a more robust transformation.

        __createCorrespondingPointsArray():
            Create matched arrays of points in the world and camera coordinate systems to estimate the transformation matrix.
    """

    def __init__(self, arrCameraMatrix, arrDistortionCoeffs=None):
        """
        :param arrCameraMatrix: A numpy array representing the intrinsic camera matrix,
                                defining the internal parameters of the camera.

        :param arrDistortionCoeffs: A numpy array or list representing the distortion coefficients of the camera.
                                    Defaults to [0, 0, 0, 0, 0] if not provided.
        """

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
        :param image: Input image used for calibration.
        :param dictWorldPoints: A dictionary containing world points mapped to their identifiers.
        :return: Transformation matrix representing the camera-to-world transformation.
        """

        ## Save as attributes
        self.calibrationImage = image
        self.dictWorldPoints = dictWorldPoints

        ## 1. Detect board and save corners/ids
        self.arrChArUcoCorners, self.arrChArUcoIds = self.oBoardDetector.detectBoard(image)
        if self.arrChArUcoCorners is None or self.arrChArUcoIds is None:
            raise Exception("No board was found")

        ## 2. Estimate the board pose w.r.t. camera
        # Calculates transformation needed to go from board coordinates to camera coordinates
        logger.debug("Estimating board pose")
        self.__estimateBoardPose()

        # cv2.drawFrameAxes(self.calibrationImage, self.arrCameraMatrix, self.arrDistortionCoeffs, self.rvec_bc, self.tvec_bc, length=50)
        # cv2.imshow("Pose Estimation", self.calibrationImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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
        Displays the detected ChArUco board on the calibration image if calibration has been performed.

        :return: Annotated image showing the detected ChArUco corners and IDs if calibration is done successfully, otherwise None.
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
        Saves the camera coordinates of ChArUco markers to a text file if calibration has been completed.

        The coordinates for each ChArUco marker are written to a file named "charuco_camera_coordinates.txt".
        Each line in the file contains the ChArUco ID and its corresponding camera coordinates.

        :raises Warning: If calibration has not been performed prior to calling this method.
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
        Estimates the board pose by computing the transformation from the board's coordinate space
        to the camera's coordinate space.

        :return: A tuple containing the rotation vector (rvec) and translation vector (tvec)
                    representing the pose of the board in the camera coordinate system.
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
        Calibrates the camera world transformation by estimating the affine 3D transformation matrix.

        This method computes the transformation matrix that converts camera coordinates to world coordinates using
        corresponding points in both spaces. It utilizes OpenCV's estimateAffine3D function to perform the computation.
        Inliers are also computed but not returned.

        :return: The 4x4 transformation matrix representing the transformation from camera to world coordinates.
        """

        # print(self.arrCamPoints)
        # print(self.arrWorldPoints)

        _, out, inliers = cv2.estimateAffine3D(self.arrCamPoints, self.arrWorldPoints, True)

        trans_mat = np.vstack((out, [0, 0, 0, 1]))
        logger.debug(f"Calculated trans_mat: {trans_mat}")

        return trans_mat

    def __calculateCameraDict(self):
        """
        Calculates a dictionary mapping ChArUco marker IDs to their corresponding 3D points in camera coordinates.

        The method iterates over the detected ChArUco IDs and computes their respective 3D points in
        camera coordinates by converting the 3D board points using the `__convertBoardToCamera` method.

        The computed points are then stored in the `self.dictCameraPoints` attribute.

        :return: None
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
        :param board_point: A 3D point in the board's coordinate system, represented as a list, tuple, or numpy array.
        :return: A 3D point converted to the camera's coordinate system, represented as a numpy array.
        """
        R, _ = cv2.Rodrigues(self.rvec_bc)
        camera_point = R@np.array(board_point) + self.tvec_bc.flatten()

        return camera_point

    def __includeDepthInfo(self):
        """
        Updates the world and camera points dictionaries with included depth information.

        Add a fixed point along the z-axis with a 50 unit offset from the surface
        into the `dictWorldPoints` dictionary. Converts the same point, represented
        in board coordinates, to camera coordinates and updates the `dictCameraPoints`
        dictionary with the transformed value.

        :return: None
        """

        # TODO: Automate the included depth information + remove hardcoding
        # Point along z-axis with offset 50 from surface
        self.dictWorldPoints[100] = [0, 0, 50]

        # Same point in camera coordinates
        # = negative z-offset from origin in board coordinates transformed to camera coordinates
        # point = [225, 0, -50]
        point = [225, 37.5, -50]
        point_camera = self.__convertBoardToCamera(point)
        self.dictCameraPoints[100] = point_camera

    def __createCorrespondingPointsArray(self):
        """
        Creates corresponding points arrays for world and camera coordinates.

        :return: None
        """

        ## Reset arrays
        self.arrWorldPoints = []
        self.arrCamPoints = []

        ## Add each point with id from the world points dictionary
        # + find point with corresponding id in the camera points list and add it to the array
        for (_id, point) in self.dictWorldPoints.items():
            self.arrWorldPoints.append(point)
            self.arrCamPoints.append(self.dictCameraPoints[_id].tolist())

        ## Reshape arrays to numpy format
        self.arrWorldPoints = np.array(self.arrWorldPoints, dtype=np.float32).reshape(-1, 3)
        self.arrCamPoints = np.array(self.arrCamPoints, dtype=np.float32).reshape(-1, 3)



import numpy as np
import cv2
import logging
import open3d as o3d

logger = logging.getLogger("Calibration")


def drawDetectedCorners(image: np.ndarray, charuco_corners: list[tuple[int, int]],
                        charuco_ids: list[int]) -> np.ndarray:
    """
    Annotating the detected ChArUco corners on the input image.

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


def annotate_image(image: np.ndarray, arrCamPoints: np.ndarray, arrWorldPoints: np.ndarray) -> np.ndarray:
    """
    Annotates an image with 2D camera points and their corresponding 3D world coordinates.

    :param image: The input image to be annotated.
    :param arrCamPoints: A numpy array of shape (N,2) containing 2D camera coordinates (u, v).
    :param arrWorldPoints: A numpy array of shape (N,3) containing corresponding 3D world coordinates (x, y, z).
    """
    annotated_image = image.copy()
    text_offset = 20  # Offset for text placement to prevent overlap

    for i, (cam_point, world_point) in enumerate(zip(arrCamPoints, arrWorldPoints)):
        x2d, y2d = int(cam_point[0]), int(cam_point[1])
        x3d, y3d, z3d = world_point

        # Draw circle on 2D camera coordinates
        cv2.circle(annotated_image, (x2d, y2d), radius=6, color=(0, 255, 0), thickness=-1)

        # Adjust text placement to reduce overlap
        text_y_offset = y2d - 10 - (i % 3) * text_offset

        # Display 3D coordinates above the point in white for better visibility
        text = f"({x3d:.1f}, {y3d:.1f}, {z3d:.1f})"
        cv2.putText(annotated_image, text, (x2d - 50, text_y_offset),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return annotated_image


class BoardDetector:
    """
    Class that handles the board detection
    """

    def __init__(self, iSquareLength: float, iMarkerLength: float, size: tuple[int, int]):
        """
        :param iSquareLength: The length of the squares on the ChArUco board, specified as a float or an int.
        :param iMarkerLength: The length of the markers on the ChArUco board, specified as a float or an int.
        :param size: A tuple of two integers representing the number of squares along the board's width and height.
        """

        ## Define the charuco board
        self.oArucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.oChArUcoBoard = cv2.aruco.CharucoBoard(
            size=size,
            markerLength=iMarkerLength,
            squareLength=iSquareLength,
            dictionary=self.oArucoDict
        )

        ## Initialize the detector object
        self.oDetector = cv2.aruco.CharucoDetector(
            self.oChArUcoBoard
        )

    def detectBoard(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Detect corners and ids of the charuco board in the input image.

        :param image: The input image in which the board is to be detected.
        :return: A tuple containing detected charuco corners and their corresponding ids.
                    Returns (None, None) if no board is detected.
        """

        ## Find corners and corresponding ids from the board
        charuco_corners, charuco_ids, _, _ = self.oDetector.detectBoard(image=image)

        if charuco_corners is None or charuco_ids is None:
            logger.error("No board was found")
            return None, None  # No board detected

        return charuco_corners, charuco_ids


class CameraCalibrator:
    def __init__(self, arrCameraMatrix: np.ndarray, arrDistortionCoeffs=None | np.ndarray):
        """
        :param arrCameraMatrix: A numpy array representing the intrinsic camera matrix,
                                defining the internal parameters of the camera.

        :param arrDistortionCoeffs: A numpy array or list representing the distortion coefficients of the camera.
                                    Defaults to [0, 0, 0, 0, 0] if not provided.
        """

        if arrDistortionCoeffs is None:
            arrDistortionCoeffs = [0, 0, 0, 0, 0]
        # logger.debug("CameraCalibrator initializing")

        ## Load intrinsics + distortion coefficients as numpy arrays
        self.arrCameraMatrix = np.asarray(arrCameraMatrix)
        self.arrDistortionCoeffs = np.asarray(arrDistortionCoeffs)

        ## Initialize the board detector (TODO: Remove hardcoded board parameters)
        self.oBoardDetector = BoardDetector(
            iSquareLength=37.5,
            iMarkerLength=25,
            size=(7, 5),
        )

        ## Initialize World-Camera transformation vectors
        self.rvec_wc = None
        self.tvec_wc = None

        ## Initialize empty pointcloud
        self.pcd = None

        ## Calibrated flag
        self.bCalibrated = False

        ## Image with board used for calibration
        self.calibrationImage = None

        ## Save detected charuco corners and ids (in image coordinates [pixels])
        self.arrChArUcoCorners = None
        self.arrChArUcoIds = None

        ## Dictionary of point coordinates with corresponding id as key
        self.dictWorldPoints = {}  # In world coordinates [3D] = (x, y, z)
        self.dictCameraImagePoints = {}  # In camera coordinates [2D] = (u, v)
        self.dictCamera3DPoints = {}  # In camera coordinates [3D] = (x, y, z)

        ## Array of points in each coordinate space to estimate transformation
        # Point in one array should correspond to the point with the same index in the other array
        self.arrWorldPoints = []
        self.arrCamPoints = []

    def runCalibration(self, image: np.ndarray, pointcloud: o3d.geometry.PointCloud, dictWorldPoints: dict) -> tuple[
        np.ndarray, float]:
        """
        Runs the calibration algorithm on the provided arguments, returns the transformation matrix and scale factor.

        :param pointcloud: Input pointcloud used for calibration.
        :param image: Input image used for calibration (should be from same timestamp as pointcloud).
        :param dictWorldPoints: A dictionary containing world points mapped to their identifiers ((ch)aruco corner ids.)
        :return: Transformation matrix + scale factor representing the camera-to-world transformation.
        """

        logger.info("Starting calibration procedure")

        ## Save as attributes
        self.calibrationImage = image
        self.dictWorldPoints = dictWorldPoints
        self.pcd = pointcloud

        ## 1. Detect board and save corners/ids
        self.arrChArUcoCorners, self.arrChArUcoIds = self.oBoardDetector.detectBoard(image)
        if self.arrChArUcoCorners is None or self.arrChArUcoIds is None:
            raise Exception("Board was not found")

        logger.info("Board detected, continuing calibration procedure")

        # Refine corners to subpixel accuracy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(
            gray,
            self.arrChArUcoCorners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        # print(self.arrChArUcoCorners)

        ## 3. Create corresponding points array
        self._createCorrespondingPointsArray()

        # ## 4. Calculate the extrinsics by solving 3D points to projection points
        # Returns transformation from camera --> world
        logger.info("Trying to calculate the transformation")

        _, rvec, tvec = cv2.solvePnP(self.arrWorldPoints, self.arrCamPoints, self.arrCameraMatrix, None)

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        trans_mat = np.eye(4)
        trans_mat[:3, :3] = R
        trans_mat[:3, 3] = t.flatten()
        trans_mat = np.linalg.inv(trans_mat)

        logger.info("Transformation found")

        self.bCalibrated = True

        return trans_mat, 1

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

    def _createCorrespondingPointsArray(self):
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
            self.arrCamPoints.append(self.arrChArUcoCorners[_id].tolist())

        ## Reshape arrays to numpy format
        self.arrWorldPoints = np.array(self.arrWorldPoints, dtype=np.float32).reshape(-1, 3)
        self.arrCamPoints = np.array(self.arrCamPoints, dtype=np.float32).reshape(-1, 2)

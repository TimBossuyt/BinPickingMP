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


def annotate_image(image, arrCamPoints, arrWorldPoints):
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

        ## Image with board used for calibration
        self.calibrationImage = None

        ## Save detected charuco corners and ids (in image coordinates [pixels])
        self.arrChArUcoCorners = None
        self.arrChArUcoIds = None

        ## Dictionary of point coordinates with corresponding id as key
        self.dictWorldPoints = {} # In world coordinates [3D] = (x, y, z)
        self.dictCameraPoints = {} # In camera coordinates [2D] = (u, v)

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



        # print(self.arrChArUcoCorners)
        # print(self.arrChArUcoIds)

        ## 2. Save detected corners as dictionary
        self.dictCameraPoints = {}
        for i, charuco_id in enumerate(self.arrChArUcoIds.flatten()):
            board_point = self.arrChArUcoCorners[i].flatten()
            self.dictCameraPoints[charuco_id] = np.asarray(board_point)

        self.annotate_and_display()

        ## 3. Create corresponding points array
        self.__createCorrespondingPointsArray()
        # print(self.arrCamPoints)
        # print(self.arrWorldPoints)

        img_annot = annotate_image(self.calibrationImage, self.arrCamPoints, self.arrWorldPoints)
        cv2.imshow("t", img_annot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        ## 4. Calculate the extrinsics by solving 3D points to projection points
        # Returns transformation from world --> camera
        _, rvec, tvec = cv2.solvePnP(
                    objectPoints=np.asarray(self.arrWorldPoints),
                    imagePoints=np.asarray(self.arrCamPoints),
                    cameraMatrix=self.arrCameraMatrix,
                    distCoeffs=self.arrDistortionCoeffs,
                    flags=cv2.SOLVEPNP_IPPE
        )

        ## Draw origin
        img_axis = cv2.drawFrameAxes(image, self.arrCameraMatrix, (0, 0, 0, 0, 0), rvec, tvec, 100)
        cv2.imshow("Annotated Image", img_axis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        R, _ = cv2.Rodrigues(rvec)

        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R
        T_w2c[:3, 3] = tvec.flatten()

        # print(T_w2c)

        self.bCalibrated = True

        ## 5. Calculate inverse
        T_c2w = np.linalg.inv(T_w2c)

        return T_c2w


    def annotate_and_display(self):
        # Make a copy of the image to avoid modifying the original
        annotated_image = self.calibrationImage.copy()

        # Iterate over detected points and draw them
        for charuco_id, (x, y) in self.dictCameraPoints.items():
            # Convert coordinates to integers
            x, y = int(x), int(y)

            # Draw a small circle at each detected corner
            cv2.circle(annotated_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

            # Put the ID near the point
            cv2.putText(annotated_image, str(charuco_id), (x + 10, y - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 0, 0), thickness=2)

        # Display the annotated image
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

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
        self.arrCamPoints = np.array(self.arrCamPoints, dtype=np.float32).reshape(-1, 2)



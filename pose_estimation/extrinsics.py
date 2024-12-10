import cv2
import numpy as np

## To-Do: Use multiple aruco markers

class ExtrinsicTransformationAruco:
    def __init__(self, pcdScene, iHeightImage, iWidthImage, oArUcoDict):
        """
        Initialize the ExtrinsicTransformationAruco object.

        Parameters:
            pcdScene: Point cloud scene containing ArUco markers
            iHeightImage (int): Height of the image
            iWidthImage (int): Width of the image
            oArUcoDict: ArUco dictionary for marker detection
        """

        self.pcdScene = pcdScene

        ## Load points as 2D array
        self.arrPoints = np.asarray(self.pcdScene.points).reshape(iHeightImage, iWidthImage, 3)

        ## Load colors as 2D array to use as image for Aruco Detection
        self.arrColors = np.asarray(self.pcdScene.colors).reshape(iHeightImage, iWidthImage, 3)  # (1920, 1080, 3), RGB
        self.arrColors = (self.arrColors * 255).astype(np.uint8)
        self.arrColorsBGR = cv2.cvtColor(self.arrColors, cv2.COLOR_RGB2BGR)

        self.oArUcoDict = oArUcoDict
        self.parameters = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(self.oArUcoDict, self.parameters)

        self.arrCamPoints = self.getCameraPointsWithDepth()

        self.arrWorldPoints = self.getWorldPointsWithDepth(True)

        self.arrTransMat = self.estimateTransformation()


    def estimateTransformation(self):
        """
        Estimate the transformation matrix between camera and world coordinates.

        Returns:
            np.array: 4x4 transformation matrix
        """
        # force_rotation=true --> to just get a rigid transformation

        retval, out, inliers = cv2.estimateAffine3D(self.arrCamPoints, self.arrWorldPoints, True)

        trans_mat = np.vstack((out, [0, 0, 0, 1]))

        return trans_mat


    def getCameraPointsWithDepth(self):
        """
        Get camera points with depth information.

        Returns:
            np.array: Array of camera points with depth
        """

        ## Getting corners from ID0
        #0: Y-axis
        #1: Diagonal
        #2: X-axis
        #3: Origin
        arrCamPoints = self.getCamCornersByID(0)

        ## Switch corner id 1 with point perpendicular with the plane to get depth information
        point_z = arrCamPoints[1] + 50 * self.getPlaneNormalUnit()

        arrCamPoints[1] = point_z

        return arrCamPoints

    def getWorldPointsWithDepth(self, bInvertDepth):
        """
        Get world points with depth information.

        Parameters:
            bInvertDepth (bool): Whether to invert the depth direction

        Returns:
            np.array: Array of world points with depth
        """

        iDir = 1
        if bInvertDepth:
            iDir = -1

        arrWorldPoints = np.array([
            [(self.arrCamPoints[0, 0] - self.arrCamPoints[3, 0]),
             (self.arrCamPoints[0, 1] - self.arrCamPoints[3, 1]),
             0],
            [(self.arrCamPoints[1, 0] - self.arrCamPoints[3, 0]),
             (self.arrCamPoints[1, 1] - self.arrCamPoints[3, 1]),
             (50*iDir)],
            [(self.arrCamPoints[2, 0] - self.arrCamPoints[3, 0]),
             (self.arrCamPoints[2, 1] - self.arrCamPoints[3, 1]),
             0],
            [0, 0, 0]
        ])

        print("ArUco markers in world coordinates")
        print(arrWorldPoints)

        return arrWorldPoints

    def getPlaneNormalUnit(self, iDistanceThreshold=10, iRansacN=100, iNumIter=1000, iProb=0.999):
        """
        Get the unit normal vector of the plane in the point cloud.

        Parameters:
            iDistanceThreshold (int): Distance threshold for RANSAC
            iRansacN (int): Number of random samples for RANSAC
            iNumIter (int): Number of iterations for RANSAC
            iProb (float): Probability of success for RANSAC

        Returns:
            np.array: Unit normal vector of the plane
        """
        oFittedPlane, arrInliersIndex = self.pcdScene.segment_plane(
            distance_threshold=iDistanceThreshold,
            ransac_n=iRansacN,
            num_iterations=iNumIter,
            probability=iProb
        )

        arrPlaneNormal = np.array(oFittedPlane[:3])
        arrPlaneNormalUnit = arrPlaneNormal / np.linalg.norm(arrPlaneNormal)

        return arrPlaneNormalUnit


    def getCamCornersByID(self, iID):
        """
        Get camera corners for a specific ArUco marker ID.

        Parameters:
            iID (int): ID of the ArUco marker

        Returns:
            np.array: Array of camera corners for the specified ArUco marker
        """

        markerCorners, markerIds, _ = self.detector.detectMarkers(self.arrColorsBGR)

        image_with_markers = cv2.aruco.drawDetectedMarkers(self.arrColorsBGR.copy(), markerCorners, markerIds)
        cv2.imshow("Detected ArUco Markers", cv2.resize(image_with_markers, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Convert to numpy arrays
        markerCorners = np.asarray(markerCorners)
        markerIds = np.asarray(markerIds)

        index_id = np.where(markerIds.flatten() == iID)[0]

        if index_id.size > 0:
            corners_id = markerCorners[index_id[0]][0]
        else:
            corners_id = None
            print(f"ArUco with ID {iID} not found")

        ## Getting 4 corners in camera coordinates [X, Y, Z]
        arrCamPoints = []
        for corner in corners_id.astype(int):
            pointCam = self.arrPoints[corner[1], corner[0], :] ## [y, x, :] for array indexing (row, column)

            ## If depth value = 0, it is not a valid point
            if pointCam[2] == 0:
                print(f"Invalid AruCo corner found")

            arrCamPoints.append(pointCam)

        arrCamPoints = np.array(arrCamPoints)

        print("ArUco markers in camera coordinates")
        print(arrCamPoints)

        return arrCamPoints

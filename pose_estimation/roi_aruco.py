import numpy as np
import cv2


def arucoImage(cvImage, markerCorners, markerIds):
    ## LLM Generated ##
    """
    Improve annotation visibility:
    - Increase marker outline thickness
    - Display larger and bolder marker IDs with a background box
    """

    image_with_markers = cvImage.copy()

    # Draw default markers
    cv2.aruco.drawDetectedMarkers(image_with_markers, markerCorners, markerIds)

    # Make marker IDs more visible with a background
    if markerIds is not None:
        for i, corner in enumerate(markerCorners):
            corner = corner.reshape(4, 2)
            top_left = tuple(map(int, corner[0]))  # Convert to integer coordinates

            marker_id = str(markerIds[i][0])  # Extract marker ID
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2  # Larger text
            thickness = 4  # Thicker lines
            text_color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(marker_id, font, font_scale, thickness)

            # Define text background rectangle
            rect_top_left = (top_left[0], top_left[1] - text_height - 10)
            rect_bottom_right = (top_left[0] + text_width + 20, top_left[1])

            # Draw filled rectangle as background
            cv2.rectangle(image_with_markers, rect_top_left, rect_bottom_right, bg_color, -1)

            # Place text on top of the background
            text_position = (top_left[0] + 10, top_left[1] - 10)
            cv2.putText(image_with_markers, marker_id, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Draw thicker marker borders
            cv2.polylines(image_with_markers, [corner.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=5)

        return image_with_markers

class ExtrinsicTransformation:
    def __init__(self, oArucoDict, iHeightImage, iWidthImage):
        ## OpenCV AruCo setup
        self.oArucoDict = oArucoDict
        self.parameters = cv2.aruco.DetectorParameters()
        self.oArucoDetector = cv2.aruco.ArucoDetector(self.oArucoDict, self.parameters)

        self.iHeightImage = iHeightImage
        self.iWidthImage = iWidthImage

        self.arrPoints = None
        self.arrColors = None
        self.arrColorsBGR = None


    def estimateWorldTransformation(self, pcdScene):
        ## Load and save pointcloud as 3D numpy arrays
        self.arrPoints = np.asarray(pcdScene.points).reshape(self.iHeightImage, self.iWidthImage, 3)

        self.arrColors = np.asarray(pcdScene.colors).reshape(self.iHeightImage, self.iWidthImage, 3)
        self.arrColorsBGR = cv2.cvtColor(self.arrColors, cv2.COLOR_RGB2BGR)

        ## Find 4 positions in camera coordinates
        # 3 AruCo + 1 depth from dominant plane
        arrCamPoints = self.getCameraPointsWithDepth()

        ## Find 4 corresponding points in world coordinates
        pass

    def getCameraPointsWithDepth(self):
        pass

    def showMarkers(self, cvImage):
        markerCorners, markerIds, _ = self.oArucoDetector.detectMarkers(cvImage)

        image_with_markers = arucoImage(cvImage, markerCorners, markerIds)

        return image_with_markers

    def __findMarkers(self):
        ## Find markers with OpenCV
        markerCorners, markerIds, _ = self.oArucoDetector.detectMarkers(self.arrColorsBGR)

        ## Show found markers (Debugging purposes)


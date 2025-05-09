import os
import numpy as np
import cv2

# ------------------------------
# PARAMETERS:
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
SQUARE_LENGTH = 0.030 ## [m]
MARKER_LENGTH = 0.020
SIZE = (7, 5) ## (vertical, horizontal)
# ------------------------------

def create_and_save_new_board():
    oChArUcoBoard = cv2.aruco.CharucoBoard(
        size=SIZE,
        markerLength=MARKER_LENGTH,
        squareLength=SQUARE_LENGTH,
        dictionary=ARUCO_DICT
    )

    imBoard = oChArUcoBoard.generateImage((3500, 2500), marginSize=100)
    cv2.imwrite("../../ArucoMarkers/charuco_board.png", imBoard)
    cv2.imshow("ChAruCo", imBoard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    create_and_save_new_board()
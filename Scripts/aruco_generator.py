
########## Generating ARUCO markers ##########
##### DICT_NxN_M #####
## NxN = 2D bit size e.g. 6x6 has a total of 36 bits
## M = grid size, number of IDs that can be generated with that dictionary

## Max 4 unique IDs

##### Ideal settings #####
## 1. Low number of unique IDs that need to be generated and read
## 2. High-quality image input containing ArUco markers
## 3. Larger NxN grid size, balanced with low number of unique ArUcoIDs,
# such that the inter-marker distance can be used to correct misread markers


import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

iMarkerID = 3
iMarkerSize = 200

## Generate marker image
oMarkerImage = cv2.aruco.generateImageMarker(aruco_dict, iMarkerID, iMarkerSize)

cv2.imwrite(f"../ArucoMarkers/ArUco_6x6_50_{iMarkerID}", oMarkerImage)

# Display the marker
cv2.imshow("ArUco Marker", oMarkerImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

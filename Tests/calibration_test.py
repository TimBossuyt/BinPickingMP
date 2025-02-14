import numpy as np
from luxonis_camera.calibrate import CameraCalibrator
import cv2

## ---------------------------------------
## Camera parameters (for 4K resolution)
arrIntrinsics4K = np.asarray([[3082.5126953125, 0.0, 1898.798583984375],
                            [0.0, 3080.830078125, 1104.401123046875],
                            [0.0, 0.0, 1.0]])


## Adjust intrinsics to new size: 1920/1080 --> first and second row / 2
arrIntrinsics = arrIntrinsics4K.copy()
arrIntrinsics[:2, :] /= 2

## OpenCV distortion array format: [k1,k2,p1,p2,k3]
## Is already distorted from the camera node inside the camera pipeline
arrDistortion = np.asarray([0, 0, 0, 0, 0])
## ---------------------------------------
oCalibrator = CameraCalibrator(arrIntrinsics, arrDistortion)

## Calibration points (manual)
charuco_3D_world = {0: [0, 0, 0],
                    5: [187.5, 0, 0],
                    23: [187.5, 112.5, 0],
                    18: [0, 112.5, 0] }


image = cv2.imread("2025-02-14_08-51-18.jpg")
trans_mat = oCalibrator.runCalibration(image, charuco_3D_world)
oCalibrator.saveCornerCameraCoordinates()

img_annot = oCalibrator.showDetectedBoard()
cv2.imshow("Window", img_annot)
print(trans_mat)
cv2.waitKey(0)
cv2.destroyAllWindows()


def transform_point(point, trans_mat):
    # Convert the 3D point to homogeneous coordinates (add a 1 for the w-coordinate)
    point_homogeneous = np.array([point[0], point[1], point[2], 1])

    # Perform matrix multiplication: transformed_point = trans_mat * point_homogeneous
    transformed_point_homogeneous = np.dot(trans_mat, point_homogeneous)

    # Return the transformed point by converting back to 3D coordinates
    return transformed_point_homogeneous[:3]  # We discard the w-coordinate

## ---------------------------------------
## Verification
point_cam = oCalibrator.dictCameraPoints[15]
point_world_gt = [112.5, 75, 0]

point_world_estim = transform_point(point_cam, trans_mat)

print(f"GT: {point_world_gt}")
print(f"Estimated: {point_world_estim}")
## ---------------------------------------
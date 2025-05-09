import depthai as dai
from pathlib import Path

"""
Read camera parameters from connected camera
- Intrinsics matrix
- Distortion matrix
"""

## Connect with device to read camera intrinsics and distortion
with dai.Device() as device:
  ## Read calibration data
  calibData = device.readCalibration()

  intrinsics4K = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)

  intrinsics1080p = calibData.getCameraIntrinsics(
    cameraId=dai.CameraBoardSocket.CAM_A,
    resizeWidth=1920,
    resizeHeight=1080,
    keepAspectRatio=True
  )


  print(intrinsics4K)
  print(intrinsics1080p)

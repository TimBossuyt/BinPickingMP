import depthai as dai

"""
Read camera parameters from connected camera
- Intrinsics matrix
- Distortion matrix
"""

## Connect with device to read camera intrinsics and distortion
with dai.Device() as device:
  ## Read calibration data
  calibData = device.readCalibration()

  intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
  distortion = calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)

  ## Print results to terminal
  print(intrinsics)
  print(distortion)


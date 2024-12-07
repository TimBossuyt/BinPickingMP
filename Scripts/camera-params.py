import depthai as dai

with dai.Device() as device:
  calibData = device.readCalibration()
  intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)

  distortion = calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)

  print(intrinsics)

  print(distortion)


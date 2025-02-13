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

  eeprom_json = calibData.eepromToJsonFile(Path("params_default.json"))

  # intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
  #
  # distortion = calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
  #
  # eeprom = calibData.getEepromData()
  #
  # ## Print results to terminal
  # print(intrinsics)
  # print(distortion)
  #
  # print(f"EEPROM: {eeprom.boardConf}")


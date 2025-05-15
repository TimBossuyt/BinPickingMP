import depthai as dai
from pathlib import Path
from datetime import datetime
from deepdiff import DeepDiff
import json

"""
Read camera calibration data from connected camera
"""

timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")

current_path = Path(f"./calibration_current.json")
factory_path = Path(f"./calibration_factory.json")

## Connect with device to read camera intrinsics and distortion
with dai.Device() as device:
  ## Read current calibration data
  calibData = device.readCalibration()
  factoryData = device.readFactoryCalibration()

  ## Save as .json files
  calibData.eepromToJsonFile(current_path)
  factoryData.eepromToJsonFile(factory_path)


# Load the JSON files
with current_path.open('r') as f:
    current_data = json.load(f)

with factory_path.open('r') as f:
    factory_data = json.load(f)

# Compare the data
diff = DeepDiff(factory_data, current_data, ignore_order=True)

# Output the result
if diff:
    print("Differences found between factory and current calibration:")
    print(diff.to_json(indent=2))
else:
    print("No differences found. Calibration parameters are identical.")





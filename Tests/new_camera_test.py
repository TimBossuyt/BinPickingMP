import logging.config
import datetime
from luxonis_camera import CameraV2, getConnectedDevices

########## Logging setup ##########
## Generate ISO 8601 timestamped filename
log_filename = datetime.datetime.now().strftime("log_%Y-%m-%dT%H-%M-%S.log")

## Read config file
logging.config.fileConfig("../logging.conf",
                          disable_existing_loggers=False,
                          defaults={'filename':f"logs/{log_filename}"})

logger = logging.getLogger("Main")
###################################

def main():
    sMxId = getConnectedDevices()[0]
    logger.info(f"Found devices: {sMxId}")

    oCamera = CameraV2()
    oCamera.Connect(sMxId)

if __name__ == "__main__":
    main()
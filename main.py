import cv2
import threading
import imutils
from luxonis_camera import Camera
import logging
import logging.config
import sys
import datetime

########## Logging setup ##########
## Generate ISO 8601 timestamped filename
log_filename = datetime.datetime.now().strftime("log_%Y-%m-%dT%H-%M-%S.log")

## Read config file
logging.config.fileConfig("logging.conf",
                          disable_existing_loggers=False ,
                          defaults={'filename':f"logs/{log_filename}"})

logger = logging.getLogger("Main")
###################################

oCamera = Camera(iFPS=5)
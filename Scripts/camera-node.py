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
logging.config.fileConfig("../logging.conf",
                          disable_existing_loggers=False ,
                          defaults={'filename':f"../logs/{log_filename}"})

logger = logging.getLogger("Main")
###################################

def preview_loop():
    cv2.namedWindow("Camera Preview", cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame = oCamera.getCvVideoPreview()  # Get the frame
            if frame is not None:
                cv2.imshow("Camera Preview", frame)  # Show the frame

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()


# Initialize camera
logger.info("Initializing camera")
oCamera = Camera(iFPS=5)
oCamera.Connect()
logger.info("Started camera thread")

# Start preview loop in a separate thread
thrPreview = threading.Thread(target=preview_loop, daemon=True)
thrPreview.start()

input("Press enter to continue...")
img = oCamera.getCvImageFrame()
oCamera.getColoredPointCloud()
cv2.imshow("Image", imutils.resize(img, width=500))

input("Press enter to exit...")
logger.info("Exit program")
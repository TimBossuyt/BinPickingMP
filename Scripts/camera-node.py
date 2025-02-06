import cv2
import threading
import imutils
from luxonis_camera import Camera
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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
oCamera = Camera(iFPS=5)
oCamera.Run()

# Start preview loop in a separate thread
thrPreview = threading.Thread(target=preview_loop, daemon=True)
thrPreview.start()

input("Press enter to exit...")
img = oCamera.getCvImageFrame()
oCamera.getColoredPointCloud()
cv2.imshow("Image", imutils.resize(img, width=500))

cv2.waitKey(0)
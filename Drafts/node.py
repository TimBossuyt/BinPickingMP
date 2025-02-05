from camera import Camera
import threading
import cv2
from flask import Flask, Response

# def main():
#     print("Running main program")
#     print("Press 'q' to exit the display window.")
#     while True:
#         # Get the current OpenCV frame
#         cvFrame = oCamera.getCvVideoPreview()
#
#         if cvFrame is not None:
#             # Display the frame in a window
#             cv2.imshow("Camera Feed", cvFrame)
#
#         # Check for user input
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):  # Exit the program
#             break
#
#     # Clean up OpenCV windows
#     cv2.destroyAllWindows()

app = Flask(__name__)

# Global camera instance
oCamera = Camera(5)

def generate_frames():
    """
    Generator function to yield frames for the MJPEG stream.
    """
    while True:
        # Get the current OpenCV frame
        cvFrame = oCamera.getCvVideoPreview()
        if cvFrame is not None:
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', cvFrame)
            frame = buffer.tobytes()

            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    """
    Route to serve the MJPEG stream.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_server():
    """
    Run the Flask server.
    """
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    ## Run camera node
    oCamera = Camera(5)

    # thMain = threading.Thread(target=main)
    thCamera = threading.Thread(target=oCamera.run)

    thCamera.start()
    # thMain.start()

    ## Wait for main thread to finish
    # thMain.join()

    try:
        run_flask_server()
    except KeyboardInterrupt:
        print("Shutting down server...")

    ## Launch stop event
    oCamera.stop()

    ## Wait for the camera thread to finish
    thCamera.join()




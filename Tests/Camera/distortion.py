import depthai as dai
import cv2

pipeline = dai.Pipeline()

# Define sources and outputs
camRgb: dai.node.Camera = pipeline.create(dai.node.Camera)

#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
# camRgb.setResolution(dai.CameraProperties.SensorResolution.THE_400_P)
# camRgb.setSize((640, 400))

# Linking
videoOut = pipeline.create(dai.node.XLinkOut)
videoOut.setStreamName("video")
camRgb.video.link(videoOut.input)

ispOut = pipeline.create(dai.node.XLinkOut)
ispOut.setStreamName("isp")
camRgb.isp.link(ispOut.input)

scale_factor = 0.5

with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    isp = device.getOutputQueue(name="isp", maxSize=1, blocking=False)

    while True:
        if video.has():
            frame = video.get().getCvFrame()
            resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.imshow("video", resized_frame)

        if isp.has():
            frame = isp.get().getCvFrame()
            resized_isp = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.imshow("isp", resized_isp)
        if cv2.waitKey(1) == ord('q'):
            break

import depthai as dai
import open3d as o3d
import numpy as np
from datetime import timedelta
import logging
import threading
import cv2
import time
import json

logger = logging.getLogger("Camera")


class CameraV2:
    def __init__(self):
        self.oPipeline = dai.Pipeline()
        self.__configurePipeline()

        ## Save pipeline as .json for debugging
        with open("pipeline_debug.json", "w") as f:
            json.dump(self.oPipeline.serializeToJson(), f, indent=4)

        ## Threading stuff
        self.thrCameraConnect = None


        ## Camera settings
        self.sMxId = None


    ## ---------- Hardware interfacing -----------
    def Disconnect(self) -> None:
        pass

    def Connect(self, sMxId) -> None:
        self.sMxId = sMxId

        try:
            logger.info(f"Connecting to camera with MxId: {sMxId}")

            ## Create threading object and start
            self.thrCameraConnect = threading.Thread(target=self.__connect)
            self.thrCameraConnect.start()
            logger.debug("Connect thread started")

        except Exception as e:
            logger.error("Error when trying to connect to camera")
            raise e

    ## -------------------------------------------
    @staticmethod
    def log_callback(log_message: dai.LogMessage):
        level = log_message.level
        # print(level)
        if level == dai.LogLevel.TRACE:
            logger.debug(f"[{log_message.level}] - {log_message.nodeIdName} - {log_message.payload}")

        if level == dai.LogLevel.INFO:
            logger.info(f"{log_message.payload}")



    ## ---------- Main camera loop -----------
    def __connect(self) -> None:
        device_info = dai.DeviceInfo(self.sMxId)
        try:
            logger.info(f"Trying to connect to camera {self.sMxId}")

            with dai.Device(self.oPipeline, device_info, dai.UsbSpeed.SUPER_PLUS) as device:
                captureQueue = device.getInputQueue('Capture')
                stillQueue = device.getOutputQueue('CapturedRGB')
                pcdQueue = device.getOutputQueue("CapturedPcd")

                device.setLogLevel(dai.LogLevel.INFO)
                device.addLogCallback(self.log_callback)
                device.setLogOutputLevel(dai.LogLevel.TRACE)
                device.setSystemInformationLoggingRate(0.1)

                ## Print the device summary
                logger.info("---------- Camera Summary ----------")
                logger.info(f"MxId: {device.getDeviceInfo().getMxId()}")
                logger.info(f"USB speed: {device.getUsbSpeed()}")
                logger.info(f"Connected cameras: {device.getConnectedCameras()}")
                logger.info("------------------------------------")

                while True:
                    user_input = input("Press enter to get image or q to quit")

                    if user_input == "q":
                        break

                    # Send capture command
                    ctrl = dai.CameraControl()
                    ctrl.setCaptureStill(True)
                    captureQueue.send(ctrl)

                    ## Wait for image to respond
                    try:
                        # Get and display image (blocking wait)
                        stillFrame = stillQueue.get()  # This will block until data is available
                        stillFrame = stillFrame.getCvFrame()  # Get the OpenCV frame from the still image

                        # Overlay FPS on image (optional)
                        resizedFrame = cv2.resize(stillFrame, None, fx=0.5, fy=0.5)
                        cv2.imshow("Captured Image", resizedFrame)
                        cv2.waitKey(1)

                    except Exception as e:
                        logger.error(f"Error capturing image: {e}")

            cv2.destroyAllWindows()

            logger.debug("Camera context manager ended")

        except Exception as e:
            logger.error(f"Error while connecting to camera {self.sMxId}: {e}")
            raise e
    ## -------------------------------------------



    ## ---------- Request functions -----------

    def getColoredPointCloud(self) -> o3d.geometry.PointCloud:
        pass

    def getCvVideoPreview(self) -> np.ndarray:
        pass

    def getCvImageFrame(self) -> np.ndarray:
        pass

    ## -------------------------------------------

    ## ---------- Configuring -----------
    def __configurePipeline(self):
        ########## CAMERAS ##########
        ## Color camera (middle) ##
        nodeCamColor = self.oPipeline.create(dai.node.ColorCamera)
        nodeCamColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        nodeCamColor.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        ## Mono Camera (left) ##
        nodeCamLeft = self.oPipeline.create(dai.node.MonoCamera)
        nodeCamLeft.setCamera("left")
        nodeCamLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        nodeCamLeft.setFps(3)

        ## Mono Camera (right) ##
        nodeCamRight = self.oPipeline.create(dai.node.MonoCamera)
        nodeCamRight.setCamera("right")
        nodeCamRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        nodeCamRight.setFps(3)

        ########## CALCULATING NODES ##########
        ## Depth node ##
        nodeDepth = self.oPipeline.create(dai.node.StereoDepth)
        nodeDepth.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        nodeDepth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        ## Pointcloud from depth-map
        nodePointCloud = self.oPipeline.create(dai.node.PointCloud)

        ########## SYNCING ##########
        ## Sync node
        nodeSync = self.oPipeline.create(dai.node.Sync)
        nodeSync.setSyncThreshold(timedelta(milliseconds=500))
        nodeSync.setSyncAttempts(-1)

        ## Output demux
        nodeDemux = self.oPipeline.create(dai.node.MessageDemux)

        ########## OUTPUTS ##########
        ## xOut node for main camera
        nodeXOutRgbCapture = self.oPipeline.create(dai.node.XLinkOut)
        nodeXOutRgbCapture.setStreamName("CapturedRGB")

        ## xOut node for pointcloud
        nodeXOutPcdCapture = self.oPipeline.create(dai.node.XLinkOut)
        nodeXOutPcdCapture.setStreamName("CapturedPcd")


        ########## INPUTS ##########
        ## xIn for getting pointcloud + rgb image
        nodeCaptureIn = self.oPipeline.create(dai.node.XLinkIn)
        nodeCaptureIn.setStreamName("Capture")

        ########## Linking ##########
        ## Mono-cameras to disparity depth calculation
        nodeCamLeft.out.link(nodeDepth.left)
        nodeCamRight.out.link(nodeDepth.right)

        ## Depth-map --> Pointcloud
        nodeDepth.depth.link(nodePointCloud.inputDepth)

        ## Still RGB + Pointcloud syncing
        nodeCamColor.still.link(nodeSync.inputs["StillImg"])
        nodePointCloud.outputPointCloud.link(nodeSync.inputs["PointCloud"])

        ## Demux sync to outputs
        nodeSync.out.link(nodeDemux.input)
        nodeDemux.outputs["StillImg"].link(nodeXOutRgbCapture.input)
        nodeDemux.outputs["PointCloud"].link(nodeXOutPcdCapture.input)

        ## Cam control in --> capture camera input
        nodeCaptureIn.out.link(nodeCamColor.inputControl)

    ## -------------------------------------------
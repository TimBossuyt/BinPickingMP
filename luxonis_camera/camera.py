import depthai as dai
from datetime import timedelta
import open3d as o3d
import threading
import logging
import queue
from pathlib import Path
import numpy as np
import json

from .calibrate import CameraCalibrator

logger = logging.getLogger("Camera")

## Defining commands
CV_IMG_REQUEST = "cv-img-request"
PCD_REQUEST = "pcd-request"

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transforms a set of 3D points using a homogeneous transformation matrix.

    :param points: (N, 3) NumPy array of 3D points.
    :param transform: (4, 4) NumPy array representing the homogeneous transformation matrix.
    :return: (N, 3) NumPy array of transformed 3D points.
    """
    if points.shape[1] != 3:
        raise ValueError("Input points should have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("Transformation matrix should have shape (4, 4)")

    # Convert points to homogeneous coordinates (N, 4)
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])

    # Apply transformation
    transformed_homogeneous = (transform @ homogeneous_points.T).T

    # Convert back to 3D coordinates
    transformed_points = transformed_homogeneous[:, :3]

    return transformed_points

def getConnectedDevices()-> list[str]:
    """
    Retrieve s the list of MX IDs for all connected DepthAI devices.

    :return: List of strings representing theMX IDs of all connected DepthAI devices.
    """

    try:
        arrDevices = []
        for device in dai.Device.getAllAvailableDevices():
              arrDevices.append(device.getMxId())

        return arrDevices
    except Exception as e:
        logger.error(f"Error while getting connected devices: {e}")
        raise e

class Camera:
    """
    Class representing a Camera and its associated functionalities.

    The `Camera` class provides methods to configure and operate a camera device,
    retrieve video preview frames, capture still image frames, and generate colored point cloud data.
    It uses threading for device connection and request processing, and maintains communication
    queues for handling inter-thread interactions.
    """
    def __init__(self, iFPS: int) -> None:
        """
        :param iFPS: Input frames per second. Specifies the desired frame rate for the camera.
        """
        self.arrCamToWorldMatrix = None
        self.CamToWorldScale = None
        self.oPipeline = dai.Pipeline()
        self.iFPS = iFPS
        self.sMxId = None

        self.bConnected = False

        self.cvVideoPreview = None

        self.thrCameraConnect = None

        ## Threading stuff
        self.evStop = threading.Event()
        # Queues for inter-thread communication
        self.request_queue = queue.Queue()
        self.response_queues = {
            CV_IMG_REQUEST: queue.Queue(),
            PCD_REQUEST: queue.Queue(),
        }

        ## Configure pipeline
        self._configurePipeline()
        ## Save pipeline as .json for debugging
        with open("./Output/pipeline_debug.json", "w") as f:
            json.dump(self.oPipeline.serializeToJson(), f, indent=4)

        self.oCalibrator = None

        self.bIsCalibrated = False

        self.imgCalibration = None

    def getColoredPointCloud(self) -> o3d.geometry.PointCloud:
        """
        Initiates a request to fetch a colored point cloud,
        waits for the response, and constructs the point cloud using the obtained points and colors.

        :return: A colored point cloud object.
        :rtype: open3d.geometry.PointCloud
        """

        try:
            ## Launch a request to the Run thread
            self.request_queue.put(PCD_REQUEST)
            logger.debug(f"Point cloud request sent to queue")

            ## Wait for a response in the dedicated response queue
            arrPoints, arrColors = self.response_queues[PCD_REQUEST].get(timeout=2)
            logger.debug(f"Point cloud response received")

            ## Create and return pointcloud once available
            ## Update if camera was calibrated
            if self.bIsCalibrated:
                arrPoints = np.asarray(arrPoints)
                ## Flip to right handed (flip y-axis)
                arrPoints[:, 1] = -arrPoints[:, 1]
                arrPoints = transform_points(arrPoints * self.CamToWorldScale, self.arrCamToWorldMatrix)

            oPointCloud = o3d.geometry.PointCloud()
            oPointCloud.points = o3d.utility.Vector3dVector(arrPoints)
            oPointCloud.colors = o3d.utility.Vector3dVector(arrColors)

            return oPointCloud

        except queue.Empty:
            logger.error("Timeout waiting for point cloud response")
        except Exception as e:
            logger.error(f"Error while getting colored point cloud: {e}")


    def getCvVideoPreview(self) -> np.ndarray:
        """
        Returns the video preview associated with the current instance.

        :return: The video preview frame.
        """

        return self.cvVideoPreview

    def getCvImageFrame(self) -> np.ndarray:
        """
        Launches a request to fetch the image frame and waits for the corresponding response.

        This method interacts with a separate thread running requests by placing a specific request
        in a request queue and retrieving the result from a corresponding
        response queue dedicated for the request type.

        :return: The image frame retrieved as a response.
        """

        try:
            ## Launch a request to the Run thread
            self.request_queue.put(CV_IMG_REQUEST)
            logger.debug("Image frame request sent to queue")

            ## Wait for a response in the dedicated response queue
            cvImg = self.response_queues[CV_IMG_REQUEST].get(timeout=2)
            logger.debug("Image frame response received")

            return cvImg

        except queue.Empty:
            logger.error("Timeout waiting for image frame response")
        except Exception as e:
            logger.error(f"Error while getting image frame: {e}")

    def Disconnect(self) -> None:
        """
        Disconnects the camera device and ensures the associated thread is properly terminated.

        This method sets the stop event to signal the camera connection thread to terminate. If the thread is still alive,
        it waits for the thread to finish execution and logs the waiting process. Once the thread has been successfully
        terminated, the camera is marked as disconnected, and the stop event is cleared.

        :return: None
        """
        try:
            self.evStop.set()
            ## Wait for thread to finish if still alive
            if self.thrCameraConnect.is_alive():
                logger.info("Waiting for connection thread to finish")
                self.thrCameraConnect.join()

            ## If thread has finished, flag not connected + reset stop event
            self.bConnected = False
            self.evStop.clear()

            logger.info("Camera disconnected successfully")

        except Exception as ex:
            logger.error(f"Error disconnecting camera: {ex}")


    def Connect(self, sMxId) -> None:
        """
        :param sMxId: Identifier for the device to be connected.
        :return: None
        """
        self.sMxId = sMxId
        try:
            logger.info(f"Connecting to camera with MxId: {sMxId}")
            ## Re-initialize object to allow multiple restarts
            self.thrCameraConnect = threading.Thread(target=self._connect)

            ## Start __connect in separate thread
            self.thrCameraConnect.start()
            logger.debug("Connect thread started")

        except Exception as ex:
            logger.error(f"Error connecting to camera: {ex}")

    def _connect(self) -> None:
        """
        Attempts to establish a connection with a camera device using the specified device MXID and pipeline.
        Upon successful connection, it initializes calibration data, intrinsics, calibrator, output queues, and
        continuously processes requests in queues until stopped. Handles video previewing, image processing,
        and point cloud generation requests.

        :return: None
        """
        device_info = dai.DeviceInfo(self.sMxId)  # MXID
        try:
            logger.info(f"Trying to connect to camera {self.sMxId}")
            with dai.Device(self.oPipeline, device_info, dai.UsbSpeed.SUPER) as device:
                ## Print the device summary
                logger.info("---------- Camera Summary ----------")
                logger.info(f"MxId: {device.getDeviceInfo().getMxId()}")
                logger.info(f"USB speed: {device.getUsbSpeed()}")
                logger.info(f"Connected cameras: {device.getConnectedCameras()}")
                logger.info("------------------------------------")

                ret = device.setIrLaserDotProjectorIntensity(1)

                if ret:
                    logger.info("Successfully turned on dot projector")
                else:
                    logger.warning("Failed to enable laser dot projector")


                ## Read and save calibration data
                calibData = device.readCalibration()
                calibData.eepromToJsonFile(Path("./CalibrationData/cam-calibration-data.json"))

                ## Read camera intrinsics for selected resolution
                intrinsics1080p = calibData.getCameraIntrinsics(
                    cameraId=dai.CameraBoardSocket.CAM_A,
                    resizeWidth=1920,
                    resizeHeight=1080,
                    keepAspectRatio=True
                )
                self.arrCameraMatrix = np.asarray(intrinsics1080p)

                self.distortion = calibData.getDistortionCoefficients(cameraId=dai.CameraBoardSocket.CAM_A)

                ## Create calibration object
                self.oCalibrator = CameraCalibrator(self.arrCameraMatrix)

                ## Create queue objects (size 1 --> always most recent information)
                qOut = device.getOutputQueue(name="out", maxSize=1,
                                             blocking=False)  # blocking False --> no pipeline freezing

                qRgbPreview = device.getOutputQueue(name="rgb", maxSize=1,
                                                    blocking=False)

                self.bConnected = True

                while not self.evStop.is_set():
                    inRgbPreview = qRgbPreview.get()

                    ## Always read video preview (smaller format)
                    cvBGRFramePreview = inRgbPreview.getCvFrame()
                    self.cvVideoPreview = cvBGRFramePreview

                    try:
                        ## Get request (FIFO) if any
                        sRequest = self.request_queue.get_nowait()

                        ## Check for correct request type
                        if sRequest == CV_IMG_REQUEST:
                            logger.debug("Image request received")

                            inMessageGroup = qOut.get()  # depthai.MessageGroup object
                            inColor = inMessageGroup["color"]  # Get message object
                            cvBGRFrame = inColor.getCvFrame()

                            ## Put response into dedicated queue
                            self.response_queues[CV_IMG_REQUEST].put(cvBGRFrame)

                        if sRequest == PCD_REQUEST:
                            logger.debug("Pointcloud request received")
                            inMessageGroup = qOut.get()  # depthai.MessageGroup object

                            inColor = inMessageGroup["color"]  # Get message object
                            inPointCloud = inMessageGroup["pcl"]  # Get message object

                            cvBGRFrame = inColor.getCvFrame()
                            arrPoints = inPointCloud.getPoints()  # numpy.ndarray[numpy.float32]
                            arrColors = cvBGRFrame.reshape(-1, 3) / 255.0

                            ## Put response into dedicated queue (as tuple)
                            self.response_queues[PCD_REQUEST].put(
                                (arrPoints, arrColors)
                            )

                    except queue.Empty:
                        ## Skip if queue is empty
                        pass

            logger.debug("Camera context manager ended")
        except Exception as e:
            logger.error(f"Error while connecting to camera {self.sMxId}: {e}")
            self.bConnected = False

    def loadCalibration(self, twc: np.ndarray, scale_twc: float):
        """
        Load known calibration data

        :param twc: 4x4 transformation matrix (world --> camera)
        :param scale_twc: scale factor (world --> camera)
        :return:
        """

        ## Set the attributes and flag as calibrated
        self.arrCamToWorldMatrix = twc
        self.CamToWorldScale = scale_twc
        self.bIsCalibrated = True

    def calibrateCamera(self, dictWorldPoints: dict[int, list[float, float, float]]) -> None:
        """
        Run the calibration algorithm (using the CameraCalibrator)

        :param dictWorldPoints: dictionary containing the XYZ coordinates of the board corners in the world frame
        :return: None, saves the calibration data inside the Camera object
        """


        ## Initialize calibrator object
        self.oCalibrator = CameraCalibrator(self.arrCameraMatrix)
        self.bIsCalibrated = False ## Don't transform the pcd from camera before calibration
        try:
            ## Request an image where board is visible
            image = self.getCvImageFrame()
            if image is None:
                logger.error("Received invalid image frame for calibration")
                return

            ## Request pointcloud
            pcd = self.getColoredPointCloud()

            ## DEBUGGING ONLY
            # o3d.visualization.draw_geometries([pcd], window_name="Pointcloud received during calibration")

            if pcd is None:
                logger.error("Received invalid pointcloud")

            ## Find the transformation matrix from the calibrator object
            trans_mat, scale = self.oCalibrator.runCalibration(image, pcd, dictWorldPoints)

        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            raise Exception("Calibration failed")

        ## <-- Assumes calibration was successful -->

        ## Save as attribute of the camera
        self.arrCamToWorldMatrix = trans_mat
        self.CamToWorldScale = scale

        ## Save as numpy files for re-use
        try:
            np.save("./CalibrationData/tcw.npy", trans_mat)
            np.save("./CalibrationData/scale_tcw.npy", scale)
        except Exception as e:
            logger.error(e)
            raise Exception("Failed to save camera calibration parameters")


        ## Flag calibration is done
        self.bIsCalibrated = True

        ## Save annotated image where corners are highlighted
        logger.debug("Saving annotated calibration image.")
        self.imgCalibration = self.oCalibrator.showDetectedBoard()

        logger.info("Calibrating successful")

    def getCalibrationImageAnnot(self):
        """
        Returns the calibration image annotation.

        :return: The calibration image annotation.
        :rtype: Object
        """

        return self.imgCalibration

    def _configurePipeline(self) -> None:
        """
        Configures a DepthAI pipeline consisting of multiple cameras, processing nodes, and output streams.
        The pipeline includes components for color image capture,
        depth calculation, point cloud generation, synchronization, and output streaming.

        :return: None
        """
        ########## CAMERAS ##########
        ## Color camera (middle) ##
        nodeCamColor = self.oPipeline.create(dai.node.ColorCamera)
        nodeCamColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        nodeCamColor.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        nodeCamColor.setFps(self.iFPS)

        ## Mono camera (left) ##
        nodeCamLeft = self.oPipeline.create(dai.node.MonoCamera)
        nodeCamLeft.setCamera("left")
        nodeCamLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        nodeCamLeft.setFps(self.iFPS)

        ## Mono camera (right) ##
        nodeCamRight = self.oPipeline.create(dai.node.MonoCamera)
        nodeCamRight.setCamera("right")
        nodeCamRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        nodeCamRight.setFps(self.iFPS)

        ########## CALCULATING NODES ##########
        ## Depth node ##
        nodeDepth = self.oPipeline.create(dai.node.StereoDepth)
        # nodeDepth.setDefaultProfilePreset(
        #     dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)  ## Preset
        nodeDepth.setDepthAlign(dai.CameraBoardSocket.CAM_A)  ## To get depth map to the same scale as the color image
        nodeDepth.setSubpixel(True)
        nodeDepth.setLeftRightCheck(True)
        nodeDepth.setExtendedDisparity(True)
        nodeDepth.setSubpixelFractionalBits(5)

        # nodeDepth.initialConfig.setDisparityShift(40)

        # nodeDepth.setPostProcessingHardwareResources(3, 3)
        medianFilter = dai.MedianFilter.MEDIAN_OFF
        nodeDepth.setMedianFilter(medianFilter)
        nodeDepth.initialConfig.PostProcessing.TemporalFilter.enabled = True

        ## Pointcloud from depth-map
        nodePointCloud = self.oPipeline.create(dai.node.PointCloud)

        ########## SYNCING ##########
        # synchronizes pointcloud frame with rgb image to match timestamps
        nodeSync = self.oPipeline.create(dai.node.Sync)
        nodeSync.setSyncThreshold(timedelta(milliseconds=50))

        ########## OUTPUTS ##########
        ## xOut node for PointCloud ##
        # send pointcloud data from OAK device to host
        nodeXOut = self.oPipeline.create(dai.node.XLinkOut)
        nodeXOut.setStreamName("out")

        ## xOut node for RGB Preview ##
        # send rgb preview data from OAK device to host
        nodeXOutRgbPreview = self.oPipeline.create(dai.node.XLinkOut)
        nodeXOutRgbPreview.setStreamName("rgb")

        ########## Linking ##########
        ## Preview
        nodeCamColor.preview.link(nodeXOutRgbPreview.input)

        ## Sync message group output to host
        nodeSync.out.link(nodeXOut.input)

        ## Depth map to pointcloud
        nodeDepth.depth.link(nodePointCloud.inputDepth)

        ## RGB + Pointcloud syncing
        nodeCamColor.isp.link(nodeSync.inputs["color"])
        nodePointCloud.outputPointCloud.link(nodeSync.inputs["pcl"])

        ## Mono-cameras to disparity depth calculation
        nodeCamLeft.out.link(nodeDepth.left)
        nodeCamRight.out.link(nodeDepth.right)
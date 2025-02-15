import depthai as dai
from datetime import timedelta
import open3d as o3d
import threading
import logging
import queue
from pathlib import Path

from .calibrate import CameraCalibrator

logger = logging.getLogger("Camera")

## Defining commands
CV_IMG_REQUEST = "cv-img-request"
PCD_REQUEST = "pcd-request"

def getConnectedDevices():
    arrDevices = []
    for device in dai.Device.getAllAvailableDevices():
          arrDevices.append(device.getMxId())

    return arrDevices

class Camera:
    """
    Represents a Camera class for managing an OAK-D Camera device, configured using DepthAI SDK.
    This includes camera connection and disconnection, data pipeline setup, calibration,
    and retrieval of captured images and pointclouds.

    Attributes:
        oPipeline: DepthAI pipeline to configure and manage the camera processing flow.
        iFPS: Integer FPS value for setting frame rates for the camera.
        sMxId: String representing the unique identifier (MXID) of the connected camera.
        bConnected: Boolean indicating whether the camera is successfully connected.
        cvVideoPreview: Last cached video preview frame from the camera.
        thrCameraConnect: Thread used to handle camera connection in a non-blocking manner.
        evStop: Threading event to safely stop the connection thread.
        request_queue: Queue for issuing requests to the camera processing thread.
        response_queues: Dictionary mapping request types (e.g., CV_IMG_REQUEST, PCD_REQUEST)
                         to their corresponding response queues.
        oCalibrator: CameraCalibrator object initialized after calibration to handle intrinsic/extrinsic parameters.
        bIsCalibrated: Boolean flag indicating whether the camera has been calibrated.
        imgCalibration: Annotated calibration image showing detected board corners.
        arrCameraMatrix: Intrinsic camera matrix obtained after connecting to device
        arrCamToWorldMatrix: Transformation matrix mapping camera points to world points.

    Methods:
        __init__(iFPS):
            Initializes the Camera object and its components, sets up threading and queues,
            and configures the DepthAI pipeline.

        getColoredPointCloud():
            Requests and retrieves a pointcloud with associated RGB data from the camera.

        getCvVideoPreview():
            Retrieves the video preview frame from the camera.

        getCvImageFrame():
            Requests and retrieves a single captured image frame from the camera.

        Disconnect():
            Safely disconnects the camera by stopping its connection thread and resetting related states.

        Connect(sMxId):
            Establishes connection to the camera specified by its MXID in a separate thread.

        __connect():
            Handles the detailed connection process to the DepthAI camera device, initializes
            queues for data exchange, retrieves calibration data, and processes requests for
            images and pointclouds in a continuously running loop.

        calibrateCamera(dictWorldPoints):
            Runs camera calibration using provided world points. Generates the transformation
            matrix and annotated image, and marks the camera as calibrated.

        getCalibrationImageAnnot():
            Returns the annotated calibration image with detected corners.

        __configurePipeline():
            Creates and configures the DepthAI pipeline, setting up color and mono cameras,
            stereo depth calculations, synchronized pointcloud generation, and XLink data streams.
    """
    def __init__(self, iFPS):
        """
        :param iFPS: Input frames per second. Specifies the desired frame rate for the camera.
        """
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
        self.__configurePipeline()

        self.oCalibrator = None

        self.bIsCalibrated = False

        self.imgCalibration = None

        ## Camera parameters
        self.arrCameraMatrix = None
        self.arrCamToWorldMatrix = None

    def getColoredPointCloud(self):
        """
        Initiates a request to fetch a colored point cloud,
        waits for the response, and constructs the point cloud using the obtained points and colors.

        :return: A colored point cloud object.
        :rtype: open3d.geometry.PointCloud
        """
        ## Launch a request to the Run thread
        self.request_queue.put(PCD_REQUEST)
        logger.debug(f"Launched {PCD_REQUEST}")

        ## Wait for a response in the dedicated response queue
        arrPoints, arrColors = self.response_queues[PCD_REQUEST].get()
        logger.debug(f"Received response for {PCD_REQUEST}")

        ## Create and return pointcloud once available
        oPointCloud = o3d.geometry.PointCloud()
        oPointCloud.points = o3d.utility.Vector3dVector(arrPoints)
        oPointCloud.colors = o3d.utility.Vector3dVector(arrColors)

        return oPointCloud

    def getCvVideoPreview(self):
        """
        Returns the video preview associated with the current instance.

        :return: The video preview object.
        :rtype: image in opencv format
        """
        return self.cvVideoPreview

    def getCvImageFrame(self):
        """
        Launches a request to fetch the image frame and waits for the corresponding response.

        This method interacts with a separate thread running requests by placing a specific request
        in a request queue and retrieving the result from a corresponding
        response queue dedicated for the request type.

        :return: The image frame retrieved as a response.
        """

        ## Launch a request to the Run thread
        self.request_queue.put(CV_IMG_REQUEST)
        logger.debug(f"Launched {CV_IMG_REQUEST}")

        ## Wait for a response in the dedicated response queue
        cvImg = self.response_queues[CV_IMG_REQUEST].get()
        logger.debug(f"Received response for {CV_IMG_REQUEST}")

        return cvImg

    def Disconnect(self):
        """
        Disconnects the camera device and ensures the associated thread is properly terminated.

        This method sets the stop event to signal the camera connection thread to terminate. If the thread is still alive,
        it waits for the thread to finish execution and logs the waiting process. Once the thread has been successfully
        terminated, the camera is marked as disconnected, and the stop event is cleared.

        :return: None
        """
        self.evStop.set()
        ## Wait for thread to finish if still alive
        if self.thrCameraConnect.is_alive():
            logger.debug("Waiting for connection to be closed")
            self.thrCameraConnect.join()

        logger.info("Camera disconnected")
        self.bConnected = False

        self.evStop.clear()

    def Connect(self, sMxId):
        """
        :param sMxId: Identifier for the device to be connected.
        :return: None
        """
        self.sMxId = sMxId

        ## Re-initialize object to allow multiple restarts
        self.thrCameraConnect = threading.Thread(target=self.__connect)

        ## Start __connect in separate thread
        self.thrCameraConnect.start()
        logger.debug("Started Connect thread")

    def __connect(self):
        """
        Tries to establish a connection with the camera device using its MXID,
        configuring necessary parameters as well as creating queues to handle image and pointcloud data requests.
        Reads camera calibration data and intrinsics for further processing,
        then continuously processes image/video requests and pointcloud data until a stop event is triggered.

        :return: Nothing
        """
        device_info = dai.DeviceInfo(self.sMxId)  # MXID
        try:
            logger.info(f"Trying to connect to camera {self.sMxId}")
            with dai.Device(self.oPipeline, device_info, dai.UsbSpeed.SUPER) as device:
                logger.info("Camera connected")

                self.bConnected = True

                ## Read and save calibration data
                calibData = device.readCalibration()
                calibData.eepromToJsonFile(Path("cam-calibration-data.json"))

                ## Read camera intrinsics for selected resolution
                intrinsics1080p = calibData.getCameraIntrinsics(
                    cameraId=dai.CameraBoardSocket.CAM_A,
                    resizeWidth=1920,
                    resizeHeight=1080,
                    keepAspectRatio=True
                )

                self.arrCameraMatrix = intrinsics1080p

                ## Create calibration object
                self.oCalibrator = CameraCalibrator(intrinsics1080p)

                ## Create queue objects
                qOut = device.getOutputQueue(name="out", maxSize=5,
                                             blocking=False)  # blocking False --> no pipeline freezing

                qRgbPreview = device.getOutputQueue(name="rgb", maxSize=5,
                                                    blocking=False)

                ## Empty output buffer
                while not self.evStop.is_set():
                    inRgbPreview = qRgbPreview.get()

                    ## Always read video preview (smaller format)
                    cvBGRFramePreview = inRgbPreview.getCvFrame()
                    self.cvVideoPreview = cvBGRFramePreview

                    try:
                        ## Get request (FIFO) if any
                        sRequest = self.request_queue.get_nowait()

                        ## TODO: https://docs.luxonis.com/software/depthai/examples/latency_measurement/

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
                        ## Skip if que is empty
                        pass

            logger.debug("Camera context manager ended")
        except Exception as e:
            logger.error(f"Failed to connect to the camera: {e}")
            self.bConnected = False

    def calibrateCamera(self, dictWorldPoints):
        """
        Calibrates the camera using a set of known world points and an input image where a calibration object
        (e.g., a checkerboard) is visible.

        This function computes the camera-to-world transformation matrix and
        updates the internal state of the object to reflect the completed calibration.

        Additionally, an annotated image with detected board corners is saved.

        :param dictWorldPoints: A dictionary containing the 3D world points of the calibration object. These points are typically pre-defined for accurate calibration.
        :return: A 4x4 transformation matrix representing the camera-to-world calibration result (homogeneous transformation matrix).
        """

        ## TODO: Add error handling
        ## Initialize calibrator object
        self.oCalibrator = CameraCalibrator(self.arrCameraMatrix)

        ## Request an image where board is visible
        image = self.getCvImageFrame()

        ## Find the transformation matrix from the calibrator object
        trans_mat = self.oCalibrator.runCalibration(image, dictWorldPoints)

        ## Save as attribute of the camera
        self.arrCamToWorldMatrix = trans_mat

        ## Flag calibration is done
        self.bIsCalibrated = True

        ## Save annotated image where corners are highlighted
        self.imgCalibration = self.oCalibrator.showDetectedBoard()

        logger.debug("Calibrating done")

        return trans_mat

    def getCalibrationImageAnnot(self):
        """
        Returns the calibration image annotation.

        :return: The calibration image annotation.
        :rtype: Object
        """

        return self.imgCalibration

    def __configurePipeline(self):
        """
        Configures a DepthAI pipeline consisting of multiple cameras, processing nodes, and output streams.
        The pipeline includes components for color image capture,
        depth calculation, point cloud generation, synchronization, and output streaming.

        :return: None
        """
        ##### Cameras #####
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

        ## Calculating nodes ##
        nodeDepth = self.oPipeline.create(dai.node.StereoDepth)
        nodeDepth.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)  ## Preset high accuracy (No median filter)

        # Link stereo mono cameras to get depth map
        nodeCamLeft.out.link(nodeDepth.left)
        nodeCamRight.out.link(nodeDepth.right)

        ## Depth map + Camera intrinsics to pointcloud ##
        nodePointCloud = self.oPipeline.create(dai.node.PointCloud)
        nodeDepth.depth.link(nodePointCloud.inputDepth)
        nodeDepth.setDepthAlign(dai.CameraBoardSocket.CAM_A)  ## To get depth map to the same scale as the color image

        ## Sync ##
        # synchronizes pointcloud frame with rgb image to match timestamps
        nodeSync = self.oPipeline.create(dai.node.Sync)
        nodeSync.setSyncThreshold(timedelta(milliseconds=50))

        nodeCamColor.isp.link(nodeSync.inputs["color"])
        nodePointCloud.outputPointCloud.link(nodeSync.inputs["pcl"])

        ## xOut node for PointCloud ##
        # send pointcloud data from OAK device to host
        nodeXOut = self.oPipeline.create(dai.node.XLinkOut)
        nodeSync.out.link(nodeXOut.input)
        nodeXOut.setStreamName("out")

        ## xOut node for RGB Preview ##
        # send rgb preview data from OAK device to host
        nodeXOutRgbPreview = self.oPipeline.create(dai.node.XLinkOut)
        nodeCamColor.preview.link(nodeXOutRgbPreview.input)
        nodeXOutRgbPreview.setStreamName("rgb")

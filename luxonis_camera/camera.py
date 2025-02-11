import depthai as dai
from datetime import timedelta
import open3d as o3d
import threading
import logging
import queue

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
    def __init__(self, iFPS):
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

    def getColoredPointCloud(self):
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
        return self.cvVideoPreview

    def getCvImageFrame(self):
        ## Launch a request to the Run thread
        self.request_queue.put(CV_IMG_REQUEST)
        logger.debug(f"Launched {CV_IMG_REQUEST}")

        ## Wait for a response in the dedicated response queue
        cvImg = self.response_queues[CV_IMG_REQUEST].get()
        logger.debug(f"Received response for {CV_IMG_REQUEST}")

        return cvImg

    def Disconnect(self):
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
        Runs the camera in a separate thread from the main thread
        """
        self.sMxId = sMxId

        ## Re-initialize object to allow multiple restarts
        self.thrCameraConnect = threading.Thread(target=self.__connect)

        ## Start __connect in separate thread
        self.thrCameraConnect.start()
        logger.debug("Started Connect thread")

    def __connect(self):
        device_info = dai.DeviceInfo(self.sMxId)  # MXID
        logger.info(f"Trying to connect to camera {self.sMxId}")
        with dai.Device(self.oPipeline, device_info, dai.UsbSpeed.SUPER) as device:
            logger.info("Camera connected")

            self.bConnected = True
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

    def __configurePipeline(self):
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

import depthai as dai
from datetime import timedelta
import open3d as o3d
import threading
import logging
import queue

logger = logging.getLogger(__name__)

## Defining commands
CV_IMG_REQUEST = "cv-img-request"
PCD_REQUEST = "pcd-request"

class Camera:
    def __init__(self, iFPS):
        self.oPipeline = dai.Pipeline()
        self.iFPS = iFPS

        ## Initialize
        self.arrPoints = None
        self.arrColors = None

        self.cvImageFrame = None
        self.cvVideoPreview = None

        ## Threading stuff
        self.evStop = threading.Event()
        self.request_queue = queue.Queue()

        self.response_queues = {
            CV_IMG_REQUEST: queue.Queue(),
            PCD_REQUEST: queue.Queue(),
        }

        ## Configure pipeline
        self._configurePipeline()

    def getColoredPointCloud(self):
        self.request_queue.put(PCD_REQUEST)
        logger.debug(f"Launched {PCD_REQUEST}")
        arrPoints, arrColors = self.response_queues[PCD_REQUEST].get()
        logger.debug(f"Received response for {PCD_REQUEST}")

        ## Create pointcloud once available
        oPointCloud = o3d.geometry.PointCloud()
        oPointCloud.points = o3d.utility.Vector3dVector(arrPoints)
        oPointCloud.colors = o3d.utility.Vector3dVector(arrColors)

        return oPointCloud

    def getCvVideoPreview(self):
        return self.cvVideoPreview

    def getCvImageFrame(self):
        self.request_queue.put(CV_IMG_REQUEST)
        logger.debug(f"Launched {CV_IMG_REQUEST}")
        cv = self.response_queues[CV_IMG_REQUEST].get()
        logger.debug(f"Received response for {CV_IMG_REQUEST}")
        return cv

    def Stop(self):
        self.evStop.set()

    def Run(self):
        """
        Runs the camera in a separate thread from the main thread
        """

        ## Start run_ in separate thread
        threading.Thread(target=self.run_, daemon=True).start()

    def run_(self):
        with dai.Device(self.oPipeline) as device:
            ## Create queue objects
            qOut = device.getOutputQueue(name="out", maxSize=5,
                                      blocking=False)  # blocking False --> no pipeline freezing

            qRgbPreview = device.getOutputQueue(name="rgb", maxSize=5,
                                         blocking=False)

            ## Empty output buffer
            while not self.evStop.is_set():
                inMessageGroup = qOut.get()  # depthai.MessageGroup object
                inRgbPreview = qRgbPreview.get()

                ## Always read video preview (smaller format)
                cvBGRFramePreview = inRgbPreview.getCvFrame()
                self.cvVideoPreview = cvBGRFramePreview

                try:
                    sRequest = self.request_queue.get_nowait()

                    if sRequest == CV_IMG_REQUEST:
                        logger.debug("Image request received")

                        inColor = inMessageGroup["color"]  # Get message object
                        cvBGRFrame = inColor.getCvFrame()

                        self.response_queues[CV_IMG_REQUEST].put(cvBGRFrame)

                    if sRequest == PCD_REQUEST:
                        logger.debug("Pointcloud request received")

                        inColor = inMessageGroup["color"]  # Get message object
                        inPointCloud = inMessageGroup["pcl"]  # Get message object

                        cvBGRFrame = inColor.getCvFrame()
                        arrPoints = inPointCloud.getPoints()  ## numpy.ndarray[numpy.float32]
                        arrColors = cvBGRFrame.reshape(-1, 3) / 255.0

                        self.response_queues[PCD_REQUEST].put(
                            (arrPoints, arrColors)
                        )

                except queue.Empty:
                    pass


    def _configurePipeline(self):
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





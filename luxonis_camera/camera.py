import threading
import depthai as dai
from datetime import timedelta
import time

class Camera:
    def __init__(self, iFPS):
        self.oPipeline = dai.Pipeline()
        self.iFPS = iFPS

        ## Initialize
        self.arrPoints = None
        self.arrColors = None

        ## Configure pipeline
        self._configurePipeline()

    def runCamera(self):
        with dai.Device(self.oPipeline) as device:
            ## Create queue object
            q = device.getOutputQueue(name="out", maxSize=50000,
                                      blocking=False)  # blocking False --> no pipeline freezing

            while True:
                inMessageGroup = q.get()  # depthai.MessageGroup object

                inColor = inMessageGroup["color"]  # Get message object
                inPointCloud = inMessageGroup["pcl"]  # Get message object

                if inColor and inPointCloud:
                    cvBGRFrame = inColor.getCvFrame()

                    self.arrPoints = inPointCloud.getPoints()  ## numpy.ndarray[numpy.float32]
                    self.arrColors = cvBGRFrame.reshape(-1, 3) / 255.0

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

        ## xOut node ##
        # used to send data from OAK device to host
        nodeXOut = self.oPipeline.create(dai.node.XLinkOut)
        nodeSync.out.link(nodeXOut.input)
        nodeXOut.setStreamName("out")

# def run():
#     while True:
#         print("Test")
#         time.sleep(2)
#
#
# oCamera = Camera(5)
#
# t1 = threading.Thread(target=oCamera.runCamera)
# t2 = threading.Thread(target=run)
# t1.start()
# t2.start()
# t1.join()
# t2.join()
#
#
# while True:
#     continue

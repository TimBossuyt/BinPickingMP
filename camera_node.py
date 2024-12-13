import depthai as dai
import open3d as o3d
from datetime import timedelta
import time
import cv2


## Parameters
FPS = 10

########## Creating pipeline and setting up the nodes ##########
pipeline = dai.Pipeline()

##### Cameras #####
## Color camera (middle) ##
nodeCamColor = pipeline.create(dai.node.ColorCamera)
nodeCamColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
nodeCamColor.setBoardSocket(dai.CameraBoardSocket.CAM_A)

## Mono camera (left) ##
nodeCamLeft = pipeline.create(dai.node.MonoCamera)
nodeCamLeft.setCamera("left")
nodeCamLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
nodeCamLeft.setFps(FPS)

## Mono camera (right) ##
nodeCamRight = pipeline.create(dai.node.MonoCamera)
nodeCamRight.setCamera("right")
nodeCamRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
nodeCamRight.setFps(FPS)

## Calculating nodes ##
nodeDepth = pipeline.create(dai.node.StereoDepth)
nodeDepth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY) ## Preset high accuracy (No median filter)

# Link stereo mono cameras to get depth map
nodeCamLeft.out.link(nodeDepth.left)
nodeCamRight.out.link(nodeDepth.right)

## Depth map + Camera intrinsics to pointcloud ##
nodePointCloud = pipeline.create(dai.node.PointCloud)
nodeDepth.depth.link(nodePointCloud.inputDepth)
nodeDepth.setDepthAlign(dai.CameraBoardSocket.CAM_A) ## To get depth map to the same scale as the color image

## Sync ##
# synchronizes pointcloud frame with rgb image to match timestamps
nodeSync = pipeline.create(dai.node.Sync)
nodeSync.setSyncThreshold(timedelta(milliseconds=50))

nodeCamColor.isp.link(nodeSync.inputs["color"])
nodePointCloud.outputPointCloud.link(nodeSync.inputs["pcl"])

## xOut node ##
# used to send data from OAK device to host
nodeXOut = pipeline.create(dai.node.XLinkOut)
nodeSync.out.link(nodeXOut.input)

nodeXOut.setStreamName("out")

## Main Loop
with dai.Device(pipeline) as device:
    isRunning = True

    ## Runs when 'Q' is pressed
    def key_callback():
        return

    ## Enable IR-grid
    # device.setIrLaserDotProjectorIntensity(1)

    ## Create queue object
    q = device.getOutputQueue(name="out", maxSize=50000, blocking=False) # blocking False --> no pipeline freezing

    ## Create visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_action_callback(81, key_callback)

    ## Create empty pointcloud object
    oPointCloud = o3d.geometry.PointCloud()

    tLastVisUpdate = time.time()
    first = True

    while isRunning:
        ## If 'Q' gets pressed --> stop while loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        inMessageGroup = q.get() # depthai.MessageGroup object

        inColor = inMessageGroup["color"] # Get message object
        inPointCloud = inMessageGroup["pcl"] # Get message object

        if inColor:
            ## Resolution: THE_1080_P --> shape (1080, 1920, 3)
            cvBGRFrame = inColor.getCvFrame()
            cv2.imshow("Color camera image - scaled", cv2.resize(cvBGRFrame, (0, 0), fx=0.5, fy=0.5))

            # print(cvBGRFrame.shape)

        if (inColor and inPointCloud) and time.time() > (tLastVisUpdate + 2):
            ## Resolution: 1280 x 800 --> shape (1024000, 3)
            arrPoints = inPointCloud.getPoints() ## numpy.ndarray[numpy.float32]
            arrColors = cvBGRFrame.reshape(-1, 3) / 255.0

            # print(arrPoints.shape)

            ## Clear pointcloud object data
            oPointCloud.clear()
            oPointCloud.points = o3d.utility.Vector3dVector(arrPoints)
            oPointCloud.colors = o3d.utility.Vector3dVector(arrColors)

            if first:
                vis.add_geometry(oPointCloud)
                first = False
            else:
                vis.update_geometry(oPointCloud)

            tLastVisUpdate = time.time()

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()













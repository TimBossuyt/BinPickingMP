import base64
from xmlrpc.server import SimpleXMLRPCServer
import logging
import cv2
import threading
import json
import time
import numpy as np

from ultralytics import SAM ## Allows model to load at the start of the program

## Custom libraries
from pose_estimation import Model, Scene, PoseEstimatorFPFH, SettingsManager, TransformVisualizer
from luxonis_camera import getConnectedDevices, Camera

logger = logging.getLogger("RPC-server")

class RpcServer:
    """
    Class to handle an XML-RPC server for managing camera operations + pose-estimation

    The server runs on a specified host and port, and exposes multiple remote procedure call (RPC) endpoints.

    Methods:
        - Run: Starts the XML-RPC server.
        - Stop: Stops the server and terminates associated threads.
        - register_methods: Registers all the RPC methods for the server.
    """
    def __init__(self, oCamera: Camera, sPoseSettingsPath: str, host="127.0.0.1", port=8005):
        """
        :param oCamera: Instance of the Camera class used to control camera operations.
        :type oCamera: Camera
        :param host: The hostname or IP address to bind the server to. Defaults to "127.0.0.1".
        :type host: str
        :param port: The port number to bind the server to. Defaults to 8005.
        :type port: int
        """

        self.oCamera = oCamera

        self.host = host
        self.port = port

        self.thrServer = threading.Thread(target=self.__run)

        self.server = SimpleXMLRPCServer((self.host, self.port), logRequests=False)
        self.__register_methods()

        ## Set up the pose estimation stuff
        self.__initializePoseEstimation(sPoseSettingsPath)

        ## Attribute initialization
        self.oScene = None
        self.dictDetectionResults = {}
        self.imgRender = None

    ################## CONFIGURATION ##################
    def __initializePoseEstimation(self, sPath: str):
        ## Load the settings
        self.SettingsManager = SettingsManager(sPath)

        ## Load model (TODO: Remove hardcoding)
        self.oModel = Model(
            sModelPath="./Input/T-stuk-filled.stl",
            settingsManager=self.SettingsManager,
            picking_pose=(20, 0, 0, 1, 0, 0)
        )

        ## Create the pose estimator object
        self.oPoseEstimator = PoseEstimatorFPFH(
            settingsManager=self.SettingsManager,
            model=self.oModel,
        )

        ## Load SAM-model
        self.oSamModel = SAM('sam2.1_b.pt')

    def Run(self):
        """
        Starts the XML-RPC server at the specified host and port.

        :return: None
        """
        logger.info(f"Initializing XML-RPC server on {self.host}:{self.port}")

        try:
            self.thrServer.start()
            logger.info("XML-RPC server thread started successfully")
        except Exception as e:
            logger.error(f"Failed to start XML-RPC server: {str(e)}")

    def __run(self):
        """
        Starts the XML-RPC server and keeps it running indefinitely.

        The method logs the initialization of the XML-RPC server and
        then begins serving requests continuously until stopped.

        :return: None
        """
        logger.info(f"Server is running on {self.host}:{self.port}")

        try:
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Error in the server loop: {str(e)}")

    def Stop(self):
        """
        Stops the XML-RPC server and performs cleanup operations.

        :return: None
        """

        logger.info("Stopping XML-RPC server")
        try:
            ## Shutdown camera thread / connection
            self.oCamera.Disconnect()
            logger.debug("Camera disconnected successfully")

            ## Stops .server_forever loop and finish all pending requests, does not close socket
            self.server.shutdown()

            ## Safely close the socket
            self.server.server_close()
            logger.info("Server socket closed")

            ## If thread still alive --> wait for it to stop
            if self.thrServer.is_alive():
                logger.debug("Waiting for server thread to finish")
                self.thrServer.join()

            logger.info("Server thread successfully terminated")
        except Exception as e:
            logger.error(f"An error occurred while stopping the server: {str(e)}")


    def __register_methods(self):
        """
        Registers all the necessary RPC methods used by the server.

        Each method corresponds to specific server-side functionality and
        is registered under the name of the respective function.

        :return: None
        """

        ## Register all required endpoints
        # handler on server side = __name__ of the function
        self.server.register_function(self.connect)
        self.server.register_function(self.getConnectedDevices)
        self.server.register_function(self.getCameraPreview)
        self.server.register_function(self.connectCameraByMxId)
        self.server.register_function(self.disconnectCamera)
        self.server.register_function(self.getImageFrame)
        self.server.register_function(self.runCalibration)
        self.server.register_function(self.getCalibrationImage)
        self.server.register_function(self.detectObjects)
        self.server.register_function(self.imgSelectedObject)
        self.server.register_function(self.imgRender)
        self.server.register_function(self.loadLastCalibration)
        self.server.register_function(self.reloadSettings)


    ################## ENDPOINTS ##################
    @staticmethod
    def connect():
        """
        Sets up and tests the connection.

        :return: Returns 0 upon successful execution.
        """

        ## Used to set up and test connection
        logger.debug("Received connect call")
        return 0

    @staticmethod
    def getConnectedDevices():
        """
        Fetches the list of currently connected devices.

        This method logs the incoming request, attempts to retrieve the connected devices,
        and logs the fetched devices. In case of any exceptions, it logs an error message.

        :return: A list of connected devices. Returns None if an exception occurs during retrieval.
        """

        logger.debug("Received getConnectedDevices call")
        try:
            devices = getConnectedDevices()
            logger.debug(f"Connected devices: {devices}")

            return devices
        except Exception as e:
            logger.error(f"Error while fetching connecting devices: {str(e)}")
            raise e

    def getCameraPreview(self):
        """
        Fetches a preview frame from the connected camera or a placeholder image
        if the camera is not connected.

        If the camera is connected and functional, it retrieves a video frame, encodes it as a JPEG image,
        and returns the encoded frame as a base64 string. If the camera is not connected or a frame is not
        available, a predefined placeholder image ('no-camera.jpg') is used instead, which is also encoded
        and returned as a base64 string.

        :return: Base64-encoded JPEG image of the camera preview or a placeholder image.
        :rtype: bytes
        """

        ## TODO: Check for problems when not connected and cvFrame is null

        # logger.debug("Received getCameraPreview call")
        try:
            if self.oCamera.bConnected:
                cvFrame = self.oCamera.getCvVideoPreview()

                if cvFrame is not None:
                    _, encoded_img = cv2.imencode('.jpg', cvFrame)
                    # noinspection PyTypeChecker
                    enc_data = base64.b64encode(encoded_img)
                    return enc_data
                else:
                    logger.warning("No frame available from connected camera")

            else:
                logger.warning("Camera is not connected, returning no-camera.jpg")
                cvFrame = cv2.imread("no-camera.jpg")
                _, encoded_img = cv2.imencode('.jpg', cvFrame)

                # noinspection PyTypeChecker
                enc_data = base64.b64encode(encoded_img)
                return enc_data

        except Exception as e:
            logger.error(f"Error while fetching camera preview: {str(e)}")

    def getImageFrame(self):
        """
        Fetches an image frame from the connected camera or a fallback image
        if the camera is not connected.

        The image frame is encoded to a base64 string.

        :return: Base64 encoded string of the image frame.
        """
        logger.info("Received request for a single camera frame")

        try:
            if self.oCamera.bConnected:
                logger.debug("Camera is connected, retrieving image frame")
                cvFrame = self.oCamera.getCvImageFrame()
                if cvFrame is not None:
                    _, buff = cv2.imencode('.jpg', cvFrame)
                    # noinspection PyTypeChecker
                    enc_data = base64.b64encode(buff)
                    return enc_data
                else:
                    logger.warning("No image frame received from the connected camera")
            else:
                logger.warning("Camera is not connected, returning no-camera.jpg")
                cvFrame = cv2.imread("no-camera.jpg")
                _, buff = cv2.imencode('.jpg', cvFrame)

                # noinspection PyTypeChecker
                enc_data = base64.b64encode(buff)
                return enc_data
        except Exception as e:
            logger.error(f"Error while fetching camera frame: {str(e)}")

    def connectCameraByMxId(self, sMxId):
        """
        :param sMxId: The identifier for the camera to connect to.
        :return: Returns 0 upon successful connection of the camera.
        """

        logger.info(f"Received request to connect camera with MxId: {sMxId}")
        try:
            self.oCamera.Connect(sMxId)
            logger.info(f"Successfully connected to camera with MxId: {sMxId}")
            return 0
        except Exception as e:
            logger.error(f"Error connecting to camera with MxId {sMxId}: {str(e)}")

    def loadLastCalibration(self):
        logger.info("Received request to load last camera calibration")

        ## 1. Read numpy files in CalibrationData
        twc = np.load("./CalibrationData/tcw.npy")
        twc_scale = float(np.load("./CalibrationData/scale_tcw.npy"))

        self.oCamera.loadCalibration(twc, twc_scale)
        logger.info("Succesfully loaded last calibration")

        return 0

    def runCalibration(self, sWorldPointsJson):
        """
        :param sWorldPointsJson: A JSON string representing the world points used for camera calibration.
        :return: An integer indicating the status of the calibration process (default: 0).
        """
        logger.info("Received request to run camera calibration")
        try:
            ## Calibrate camera function expects dictionary --> json to dict
            dictWorldPoints = self.__deserializeWorldPointsJson(sWorldPointsJson)
            logger.debug(f"World points for calibration: {dictWorldPoints}")

            self.oCamera.calibrateCamera(dictWorldPoints)
            logger.info("Camera calibration completed successfully")

            return 0
        except Exception as e:
            logger.error(f"Error during camera calibration: {str(e)}")
            return 1

    def getCalibrationImage(self):
        """
        :return: Encoded calibration image in Base64 format
        """

        logger.info("Received request for calibration image")
        try:
            img = self.oCamera.imgCalibration
            _, buff = cv2.imencode('.jpg', img)
            # noinspection PyTypeChecker
            enc_data = base64.b64encode(buff)

            return enc_data
        except Exception as e:
            logger.error(f"Error while retrieving calibration image: {str(e)}")

    def disconnectCamera(self):
        """
        Disconnects the camera.

        This method is used to disconnect the currently connected camera by calling the
        appropriate disconnect function of the camera object. Logs the call for debugging purposes.

        :return: Returns 0 upon successful disconnection.
        """

        logger.info("Received request to disconnect camera")
        try:
            self.oCamera.Disconnect()
            logger.info("Camera disconnected successfully")

            return 0
        except Exception as e:
            logger.error(f"Error while disconnecting camera: {str(e)}")
            return 1

    def detectObjects(self):
        logger.info("Received detect objects request")
        tStart = time.time()

        if not self.oCamera.bIsCalibrated:
            raise Exception("First run calibration")

        ## 1. Get pointcloud scene from camera
        pcdScene = self.oCamera.getColoredPointCloud()

        ## 2. Create scene object
        self.oScene = Scene(
            raw_pcd=pcdScene,
            settingsmanager=self.SettingsManager,
            samModel=self.oSamModel
        )

        ## 3. Find the array of transformations
        dictTransformResults = self.oPoseEstimator.findObjectTransforms(self.oScene)

        self.dictDetectionResults = dictTransformResults
        tEnd = time.time()
        logger.info(f"Complete object detection took: {(tEnd - tStart)*1000:.2f} ms")

        oTransformVisualizer = TransformVisualizer(self.oModel, self.oScene, dictTransformResults)
        oTransformVisualizer.displayFoundObjects()

        self.imgRender = oTransformVisualizer.renderFoundObjects()

        ## Dump results to json string
        # 1. Convert dict to serializable data types (e.g. no ndarray)
        dictSerializeProof = {}

        for key, values in dictTransformResults.items():
            transform = values[0].tolist()
            fitness = float(values[1])
            inlier_rmse = float(values[2])

            dictSerializeProof[int(key)] = (transform, fitness, inlier_rmse)

        # 2. Dump as JSON string
        dict_serialized = json.dumps(dictSerializeProof)

        return dict_serialized

    def imgSelectedObject(self, iId: int):
        ## 1. Get mask for selected object
        mask = self.oScene.dictMasks[iId]

        ## 2. Get image
        img = self.oScene.arrColours.copy()

        ## Define overlay
        blue_overlay = np.array([255, 0, 0], dtype=np.uint8)

        overlay = img.copy()
        overlay[mask == 1] = (1 - 0.5) * img[mask == 1] + 0.5 * blue_overlay
        overlay = overlay.astype(np.uint8)

        _, buff = cv2.imencode('.jpg', overlay)
        # noinspection PyTypeChecker
        enc_data = base64.b64encode(buff)

        return enc_data

    def imgRender(self):
        if self.imgRender is not None:
            _, buff = cv2.imencode('.jpg', self.imgRender)
            # noinspection PyTypeChecker
            enc_data = base64.b64encode(buff)

            return enc_data


    def reloadSettings(self):
        logger.info("Received request to reload the settings")
        self.SettingsManager.reload_settings()
        ## Reload pose estimator settings
        self.oPoseEstimator.oSm = self.SettingsManager
        self.oPoseEstimator.reload_settings()

        ## Reload scene settings
        self.oScene.oSm = self.SettingsManager
        self.oScene.reload_settings()

        ## Reload model settings
        self.oModel.oSm = self.SettingsManager
        self.oModel.reload_settings()

        logger.info("Successfully reloaded the settings")

        return 0


    ################## UTILS ##################
    @staticmethod
    def __deserializeWorldPointsJson(dict_serialized):
        """
        :param dict_serialized: A JSON string representing a dictionary where keys are expected to be numeric (as strings) and values are associated data.
        :return: A dictionary with keys converted to integers and values preserved from the input JSON string.
        """
        logger.debug(f"Deserializing world points JSON: {dict_serialized}")

        try:
            dict_world_points = json.loads(dict_serialized)

            dict_world_points = {int(k): v for k, v in dict_world_points.items()}

            logger.debug(f"Deserialized world points: {dict_world_points}")
            return dict_world_points
        except Exception as e:
            logger.error(f"Error deserializing world points JSON: {str(e)}")
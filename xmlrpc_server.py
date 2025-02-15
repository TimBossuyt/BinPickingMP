import base64
from xmlrpc.server import SimpleXMLRPCServer
from luxonis_camera import getConnectedDevices, Camera
import logging
import cv2
import threading
import json

logger = logging.getLogger("RPC-server")

class RpcServer:
    """
    RpcServer is a class for managing an XML-RPC server that allows interaction with a Camera object.

    It provides methods to control, retrieve data, and manage the state of the camera
    through registered XML-RPC endpoints.

    Attributes:
        oCamera (Camera): Instance of the Camera object, used to perform camera-related operations.
        host (str): Host address of the server. Defaults to "127.0.0.1".
        port (int): Port number on which the server runs. Defaults to 8005.
        thrServer (Thread): Thread responsible for running the server in the background.
        server (SimpleXMLRPCServer): XML-RPC server instance used for serving requests.

    Methods:
        Run:
            Starts the XML-RPC server in a separate thread.

        __run:
            Internal method to run the XML-RPC server and serve requests indefinitely.

        Stop:
            Gracefully stops the XML-RPC server, shuts down connections, and ensures the thread terminates.

        register_methods:
            Registers the XML-RPC endpoint functions to the server.

        connect:
            Tests the connection setup for the server.

        getConnectedDevices:
            Retrieves the list of connected devices.

        getCameraPreview:
            Returns a base64-encoded JPEG image of the camera's video preview.
            If the camera is not connected, it returns a default "no-camera" placeholder image.

        getImageFrame:
            Returns a base64-encoded JPEG image that represents the latest captured frame from the connected camera.
            If the camera is not connected, it returns a default "no-camera" placeholder image.

        connectCameraByMxId:
            Connects to a camera given its MxId identifier.

        runCalibration:
            Runs the camera calibration using a JSON-serialized dictionary of world points.

        getCalibrationImage:
            Retrieves the base64-encoded JPEG calibration image produced during camera calibration.

        disconnectCamera:
            Disconnects the currently connected camera and stops its corresponding session.

        __deserializeWorldPointsJson:
            Internal utility to deserialize a JSON string of world points into a dictionary with integer keys.
    """
    def __init__(self, oCamera: Camera, host="127.0.0.1", port=8005):
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
        self.register_methods()

    ################## CONFIGURATION ##################
    def Run(self):
        """
        Starts the XML-RPC server at the specified host and port.

        :return: None
        """
        logger.debug(f"Starting XML-RPC server on {self.host}:{self.port}")
        self.thrServer.start()

    def __run(self):
        """
        Starts the XML-RPC server and keeps it running indefinitely.

        The method logs the initialization of the XML-RPC server and
        then begins serving requests continuously until stopped.

        :return: None
        """
        logger.info("Started XML-RPC server")
        self.server.serve_forever()

    def Stop(self):
        """
        Stops the XML-RPC server and performs cleanup operations.

        :return: None
        """

        logger.info("Stopping XML-RPC server")
        ## Shutdown camera thread / connection
        self.oCamera.Disconnect()

        ## Stops .server_forever loop and finish all pending requests, does not close socket
        self.server.shutdown()

        ## Safely close the socket
        self.server.server_close()

        ## If thread still alive --> wait for it to stop
        if self.thrServer.is_alive():
            logger.debug("Waiting for server thread to finish")
            self.thrServer.join()

        logger.info("Stopped XML-RPC server")


    def register_methods(self):
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


    ################## ENDPOINTS ##################
    def connect(self):
        """
        Sets up and tests the connection.

        :return: Returns 0 upon successful execution.
        """

        ## Used to set up and test connection
        logger.debug("Received connect call")
        return 0

    def getConnectedDevices(self):
        """
        Fetches the list of currently connected devices.

        :return: A list of connected devices
        """

        logger.debug("Received getConnectedDevices call")
        return getConnectedDevices()

    def getCameraPreview(self):
        """
        Retrieves the current camera preview frame as a base64 encoded image.
        If the camera is connected, it fetches the live feed.

        If the camera is not connected, it displays a default "no-camera" image.

        :return: Base64 encoded image of the camera preview or a default image based on the camera's connection status.
        """

        ## TODO: Check for problems when not connected and cvFrame is null

        # logger.debug("Received getCameraPreview call")
        if self.oCamera.bConnected:
            cvFrame = self.oCamera.getCvVideoPreview()

            if cvFrame is not None:
                _, buff = cv2.imencode('.jpg', cvFrame)

                enc_data = base64.b64encode(buff)
                return enc_data
        else:
            cvFrame = cv2.imread("no-camera.jpg")
            _, buff = cv2.imencode('.jpg', cvFrame)

            enc_data = base64.b64encode(buff)
            return enc_data

    def getImageFrame(self):
        """
        Fetches an image frame from the connected camera or a fallback image
        if the camera is not connected.

        The image frame is encoded to a base64 string.

        :return: Base64 encoded string of the image frame.
        """

        if self.oCamera.bConnected:
            cvFrame = self.oCamera.getCvImageFrame()

            if cvFrame is not None:
                _, buff = cv2.imencode('.jpg', cvFrame)

                enc_data = base64.b64encode(buff)
                return enc_data
        else:
            cvFrame = cv2.imread("no-camera.jpg")
            _, buff = cv2.imencode('.jpg', cvFrame)

            enc_data = base64.b64encode(buff)
            return enc_data


    def connectCameraByMxId(self, sMxId):
        """
        :param sMxId: The identifier for the camera to connect to.
        :return: Returns 0 upon successful connection of the camera.
        """
        logger.debug(f"Received connectCameraByMxId ({sMxId}) call")
        self.oCamera.Connect(sMxId)
        return 0

    def runCalibration(self, sWorldPointsJson):
        """
        :param sWorldPointsJson: A JSON string representing the world points used for camera calibration.
        :return: An integer indicating the status of the calibration process (default: 0).
        """

        logger.debug(f"Received runCalibration call")

        ## Calibrate camera function expects dictionary --> json to dict
        dictWorldPoints = self.__deserializeWorldPointsJson(sWorldPointsJson)

        self.oCamera.calibrateCamera(dictWorldPoints)

        return 0

    def getCalibrationImage(self):
        """
        :return: Encoded calibration image in Base64 format
        """
        img = self.oCamera.imgCalibration

        _, buff = cv2.imencode('.jpg', img)

        enc_data = base64.b64encode(buff)

        return enc_data


    def disconnectCamera(self):
        """
        Disconnects the camera.

        This method is used to disconnect the currently connected camera by calling the
        appropriate disconnect function of the camera object. Logs the call for debugging purposes.

        :return: Returns 0 upon successful disconnection.
        """
        logger.debug(f"Received disconnectCamera call")
        self.oCamera.Disconnect()
        return 0

    ################## UTILS ##################
    def __deserializeWorldPointsJson(self, dict_serialized):
        """
        :param dict_serialized: A JSON string representing a dictionary where keys are expected to be numeric (as strings) and values are associated data.
        :return: A dictionary with keys converted to integers and values preserved from the input JSON string.
        """
        dict_world_points = json.loads(dict_serialized)

        dict_world_points = {int(k): v for k, v in dict_world_points.items()}

        return dict_world_points
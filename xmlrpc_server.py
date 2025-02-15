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
            logger.debug("Camera disconnected succesfully")

            ## Stops .server_forever loop and finish all pending requests, does not close socket
            self.server.shutdown()

            ## Safely close the socket
            self.server.server_close()
            logger.info("Server socket closed")

            ## If thread still alive --> wait for it to stop
            if self.thrServer.is_alive():
                logger.debug("Waiting for server thread to finish")
                self.thrServer.join()

            logger.info("Server thread succesfully terminated")
        except Exception as e:
            logger.error(f"An error occurred while stopping the server: {str(e)}")


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
        try:
            devices = getConnectedDevices()
            logger.debug(f"Connected devices: {devices}")

            return devices
        except Exception as e:
            logger.error(f"Error while fetching connecting devices: {str(e)}")


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
                    _, buff = cv2.imencode('.jpg', cvFrame)
                    enc_data = base64.b64encode(buff)
                    return enc_data
                else:
                    logger.warning("No frame available from connected camera")

            else:
                logger.warning("Camera is not connected, returning no-camera.jpg")
                cvFrame = cv2.imread("no-camera.jpg")
                _, buff = cv2.imencode('.jpg', cvFrame)

                enc_data = base64.b64encode(buff)
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
                    enc_data = base64.b64encode(buff)
                    return enc_data
                else:
                    logger.warning("No image frame received from the connected camera")
            else:
                logger.warning("Camera is not connected, returning no-camera.jpg")
                cvFrame = cv2.imread("no-camera.jpg")
                _, buff = cv2.imencode('.jpg', cvFrame)
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

    def getCalibrationImage(self):
        """
        :return: Encoded calibration image in Base64 format
        """

        logger.info("Received request for calibration image")
        try:
            img = self.oCamera.imgCalibration
            _, buff = cv2.imencode('.jpg', img)
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

    ################## UTILS ##################
    def __deserializeWorldPointsJson(self, dict_serialized):
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
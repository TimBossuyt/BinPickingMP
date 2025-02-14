import base64
from xmlrpc.server import SimpleXMLRPCServer
from luxonis_camera import getConnectedDevices, Camera
import logging
import cv2
import threading
import json

logger = logging.getLogger("RPC-server")

class RpcServer:
    def __init__(self, oCamera: Camera, host="127.0.0.1", port=8005):
        self.oCamera = oCamera


        self.host = host
        self.port = port

        self.thrServer = threading.Thread(target=self.__run)

        self.server = SimpleXMLRPCServer((self.host, self.port), logRequests=False)
        self.register_methods()

    ################## CONFIGURATION ##################
    def Run(self):
        logger.debug(f"Starting XML-RPC server on {self.host}:{self.port}")
        self.thrServer.start()

    def __run(self):
        logger.info("Started XML-RPC server")
        self.server.serve_forever()

    def Stop(self):
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
        ## Register all required endpoints
        # handler on server side = __name__ of the function
        self.server.register_function(self.connect)
        self.server.register_function(self.getConnectedDevices)
        self.server.register_function(self.getCameraPreview)
        self.server.register_function(self.connectCameraByMxId)
        self.server.register_function(self.disconnectCamera)
        self.server.register_function(self.getImageFrame)
        self.server.register_function(self.runCalibration)


    ################## ENDPOINTS ##################
    def connect(self):
        ## Used to set up and test connection
        logger.debug("Received connect call")
        return 0

    def getConnectedDevices(self):
        logger.debug("Received getConnectedDevices call")
        return getConnectedDevices()

    def getCameraPreview(self):
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
        logger.debug(f"Received connectCameraByMxId ({sMxId}) call")
        self.oCamera.Connect(sMxId)
        return 0

    def runCalibration(self, sWorldPointsJson):
        logger.debug(f"Received runCalibration call")

        dictWorldPoints = self.__deserializeWorldPointsJson(sWorldPointsJson)

        self.oCamera.calibrateCamera(dictWorldPoints)



        return 0

    def disconnectCamera(self):
        logger.debug(f"Received disconnectCamera call")
        self.oCamera.Disconnect()
        return 0

    ################## UTILS ##################
    def __deserializeWorldPointsJson(self, dict_serialized):
        dict_world_points = json.loads(dict_serialized)

        dict_world_points = {int(k): v for k, v in dict_world_points.items()}

        return dict_world_points
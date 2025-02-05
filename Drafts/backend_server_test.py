from luxonis_camera import Camera
from threading import Thread, Event
from xmlrpc.server import SimpleXMLRPCServer
import base64
import cv2

### Main thread: XML-RPC Server
### Second thread: Camera

oCamera = Camera(5)

class MyFuncs:
    def sum(self, x, y):
        return x + y

    def connect(self):
        return 0

    def getImage(self):

        ## Getting image
        cvImage = oCamera.getCvImageFrame()
        cv2.imwrite("test.jpg", cvImage)
        _, buffer = cv2.imencode(".jpg", cvImage)
        buffer = buffer.tobytes()
        data = base64.b64encode(buffer)

        return data

def runServer():
    # Create server
    with SimpleXMLRPCServer(('localhost', 8000)) as server:
        server.register_introspection_functions()

        # Register an instance; all the methods of the instance are
        # published as XML-RPC methods
        server.register_instance(MyFuncs())

        # Run the server's main loop
        server.serve_forever()


thrCamera = Thread(target=oCamera.run)
thrServer = Thread(target=runServer)

thrServer.start()
thrCamera.start()

thrServer.join()
thrCamera.join()






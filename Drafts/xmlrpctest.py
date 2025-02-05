from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import base64

class MyFuncs:
    def sum(self, x, y):
        return x + y

    def connect(self):
        return 0

    def getImage(self):
        with open("../image.jpg", "rb") as image_file:
            data = base64.b64encode(image_file.read())

        return data


# Create server
with SimpleXMLRPCServer(('localhost', 8000)) as server:
    server.register_introspection_functions()

    # Register an instance; all the methods of the instance are
    # published as XML-RPC methods
    server.register_instance(MyFuncs())

    # Run the server's main loop
    server.serve_forever()
import logging
import threading
from xmlrpc_server import RpcServer
import open3d as o3d
import time

logger = logging.getLogger("Pointcloud Visualizer")


class PointCloudVisualizer:
    def __init__(self, oServer: RpcServer):
        self.oServer = oServer
        self.vis = o3d.visualization.Visualizer()

    def Run(self):
        thread = threading.Thread(target=self.__runVisualizer, daemon=True)
        thread.start()

    def __runVisualizer(self):
        ## Create visualization window
        self.vis.create_window(
            window_name="Pose Estimation",
            width=1024,
            height=768,
        )

        ## Add origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        self.vis.add_geometry(origin)

        while True:
            self.vis.poll_events()
            self.vis.update_renderer()


            time.sleep(0.05)


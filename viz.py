import logging
import threading
from xmlrpc_server import RpcServer
import open3d as o3d
import time

logger = logging.getLogger("Pointcloud Visualizer")

#### !!!!! CODE DOES NOT WORK, VISUALIZER CANNOT WORK IN A SEPARATE THREAD


class PointCloudVisualizer:
    def __init__(self, oServer: RpcServer):
        self.oServer = oServer
        self.vis = o3d.visualization.Visualizer()

        self.thread = threading.Thread(target=self.__runVisualizer)
        self.evStop = threading.Event()

    def Run(self):

        self.thread.start()
        logger.info("Started the pointcloud visualization window")

    def Stop(self):
        logger.info("Closing pointcloud visualization window")
        try:
            self.evStop.set()

            ## Wait for thread to finish
            if self.thread.is_alive():
                self.thread.join()

            ## Destroy the window
            self.vis.destroy_window()

            self.evStop.clear()

            logger.info("Pointcloud visualization window closed")

        except Exception as e:
            logger.error(f"Error closing visualization window: {e}")

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

        while not self.evStop.is_set():
            self.vis.poll_events()
            self.vis.update_renderer()

            time.sleep(0.05)




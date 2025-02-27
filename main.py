from luxonis_camera import Camera
import logging.config
import datetime
from xmlrpc_server import RpcServer
import open3d as o3d
import time



########## Logging setup ##########
## Generate ISO 8601 timestamped filename
log_filename = datetime.datetime.now().strftime("log_%Y-%m-%dT%H-%M-%S.log")

## Read config file
logging.config.fileConfig("logging.conf",
                          disable_existing_loggers=False ,
                          defaults={'filename':f"logs/{log_filename}"})

logger = logging.getLogger("Main")
###################################

########## Camera setup ##########
oCamera = Camera(5)
##################################

########## Visualizer setup ##########
vis = o3d.visualization.VisualizerWithKeyCallback()
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
###################################

########## XML-RPC setup ##########
oServer = RpcServer(
    oCamera = oCamera,
    host="127.0.0.1",
    port=8005,
    sPoseSettingsPath="./pose_settings.json"
)
###################################

def main():
    oServer.Run()

    ## Visualizer
    vis.create_window(
        window_name="Pose Estimation",
        width=1024,
        height=768
    )
    vis.add_geometry(origin)

    vis.run()

    vis.destroy_window()

    logger.debug("Main thread exit command")

    logger.debug("Trying to stop server thread")
    # oVisualizer.Stop()
    oServer.Stop()

    logger.info("Everything finished nicely")

if __name__ == "__main__":
    main()




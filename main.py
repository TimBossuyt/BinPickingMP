from luxonis_camera import Camera
import logging.config
import datetime
from xmlrpc_server import RpcServer

########## Logging setup ##########
## Generate ISO 8601 timestamped filename
log_filename = datetime.datetime.now().strftime("log_%Y-%m-%dT%H-%M-%S.log")

## Read config file
logging.config.fileConfig("logging.conf",
                          disable_existing_loggers=False ,
                          defaults={'filename':f"./Output/logs/{log_filename}"})

logger = logging.getLogger("Main")
###################################

########## Camera setup ##########
oCamera = Camera(5)
##################################

########## XML-RPC setup ##########
oServer = RpcServer(
    oCamera = oCamera,
    host="127.0.0.1",
    port=8005,
    sPoseSettingsPath="./settings.json"
)
###################################

def main():
    oServer.Run()

    ## Press enter to exit main thread
    input()
    logger.debug("Main thread exit command")

    logger.debug("Trying to stop server thread")
    oServer.Stop()

    logger.info("Everything finished nicely")

if __name__ == "__main__":
    main()




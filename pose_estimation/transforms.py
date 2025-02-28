import numpy as np

## DRAFT!!

class Transformer:
    ## Defined by 3 coordinate systems
    # Cobot-coordinate system
    # Camera-coordinate system
    # CAD-coordinate system

    def __init__(self):
        ## Transforms between cobot and camera
        self.arrCobotToCam = None
        self.arrCamToCobot = None

        ## Only CAD --> cobot needed
        self.arrCADToCobot = None


    def setCameraToCobot(self, arrTransmat: np.ndarray) -> None:
        ## Save camera to cobot transformation
        self.arrCamToCobot = arrTransmat

        ## Save inverse if necessary
        #...

    def transformPoint(self):
        pass








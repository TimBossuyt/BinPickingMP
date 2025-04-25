import open3d as o3d
import logging

from .utils import filter_points_by_x_range, filter_points_by_z_range
from .settings import SettingsManager

logger = logging.getLogger("Model object")

class Model:
    """
    Manages CAD-model for pose estimation
    """

    def __init__(self, sModelPath, settingsManager: SettingsManager, picking_pose: tuple[float, float, float, float, float, float]):
        self.oSm = settingsManager
        self.__loadSettings()

        ## Load model as mesh
        self.mshModel = o3d.io.read_triangle_mesh(sModelPath)
        ## Sampling mesh to create pointcloud
        self.pcdModel = self.mshModel.sample_points_poisson_disk(number_of_points=self.iPoints)

        ## Run model preprocessing steps
        self.__optimizeModel()

        ## Picking pose = (x, y, z, NX, NY, NZ)
        self.picking_pose = picking_pose

    def getPickPosition(self):
        return self.picking_pose[0], self.picking_pose[1], self.picking_pose[2]

    def getPickNormal(self):
        return self.picking_pose[3], self.picking_pose[4], self.picking_pose[5]

    def __loadSettings(self):
        self.iNormalRadius = self.oSm.get("Model.NormalRadius")
        self.iPoints = self.oSm.get("Model.NumberOfPoints")

    def reload_settings(self):
        self.__loadSettings()

        ## Recalculate model optimizer
        logger.info("Recalculating model features")
        self.__optimizeModel()

        logger.info("Reloaded settings")


    def __optimizeModel(self):
        ## MODEL SPECIFIC!!!! TODO: Change model specific optimization
        # Only selects the upper half of the model (remove symmetry)
        self.pcdModel = filter_points_by_z_range(self.pcdModel, 0, 500)
        # self.pcdModel = filter_points_by_z_range(self.pcdModel, 0, 1000)

        ## Re-estimate the surface normals
        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iNormalRadius)
        self.pcdModel.estimate_normals(oNormalSearchParam)
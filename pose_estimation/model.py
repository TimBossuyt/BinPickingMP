import open3d as o3d

from .utils import filter_points_by_x_range
from .settings import SettingsManager

class Model:
    """
    Manages CAD-model for pose estimation
    """

    def __init__(self, sModelPath, settingsManager: SettingsManager, picking_pose):
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

    def __loadSettings(self):
        self.iNormalRadius = self.oSm.get("Model.NormalRadius")
        self.iPoints = self.oSm.get("Model.NumberOfPoints")


    def __optimizeModel(self):
        ## MODEL SPECIFIC!!!! TODO: Change model specific optimization
        # Only selects the upper half of the model (remove symmetry)
        self.pcdModel = filter_points_by_x_range(self.pcdModel, 0, 300)

        ## Re-estimate the surface normals
        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iNormalRadius)
        self.pcdModel.estimate_normals(oNormalSearchParam)
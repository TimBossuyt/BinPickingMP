import open3d as o3d
from .utils import *


class Model:
    """
    Manages CAD-model for pose estimation
    """
    def __init__(self, sModelPath, iPoints, iNormalRadius):
        self.iNormalRadius = iNormalRadius

        ## Loading model as mesh
        self.mshModel = o3d.io.read_triangle_mesh(sModelPath)

        ## Sampling mesh to create pointcloud
        self.pcdModel = self.mshModel.sample_points_poisson_disk(number_of_points=iPoints)

        ## Run custom script to optimize CAD-model pointcloud
        self.optimizeModel()

    def optimizeModel(self):
        self.pcdModel = filter_points_by_x_range(self.pcdModel, 0, 300)

        ## Re-estimate surface normals
        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iNormalRadius)
        self.pcdModel.estimate_normals(oNormalSearchParam)

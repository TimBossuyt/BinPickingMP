import open3d as o3d
from utils import filter_points_by_x_range


class Model:
    """
    Manages CAD-model for pose estimation
    """

    def __init__(self, sModelPath, iPoints, iNormalRadius, picking_pose):
        self.iNormalRadius = iNormalRadius

        ## Load model as mesh
        self.mshModel = o3d.io.read_triangle_mesh(sModelPath)

        ## Sampling mesh to create pointcloud
        self.pcdModel = self.mshModel.sample_points_poisson_disk(number_of_points=iPoints)

        ## Picking pose = (x, y, z, NX, NY, NZ)
        self.picking_pose = picking_pose

    def __optimizeModel(self):
        ## MODEL SPECIFIC!!!! TODO: Change model specific optimization
        # Only selects the upper half of the model (remove symmetry)
        self.pcdModel = filter_points_by_x_range(self.pcdModel, 0, 300)

        ## Re-estimate the surface normals
        oNormalSearchParam = o3d.geometry.KDTreeSearchParamRadius(radius=self.iNormalRadius)
        self.pcdModel.estimate_normals(oNormalSearchParam)
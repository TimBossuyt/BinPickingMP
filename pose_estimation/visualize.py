import open3d as o3d
from pathlib import Path
import cv2
import logging
import numpy as np
import copy

from .model import Model
from .scene import Scene
from .settings import SettingsManager
from .utils import display_point_clouds
from .pose_estimator import PoseEstimatorFPFH

logger = logging.getLogger("Pose Visualizer")


def create_arrow(origin, normal):
    length = 150

    ## LLM GENERATED ##
    mshArrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=3,
        cone_radius=6,
        cylinder_height=length*0.8,
        cone_height=length*0.2
    )

    # Normalize the normal vector
    normal = np.array(normal) ## Inverse to point arrow downwards
    normal = normal / np.linalg.norm(normal)  # Ensure it's a unit vector

    # Compute the rotation matrix to align the arrow with the normal vector
    z_axis = np.array([0, 0, 1])  # Default direction of Open3D arrow
    axis = np.cross(z_axis, normal)
    angle = np.arccos(np.dot(z_axis, normal))  # Angle between z_axis and normal

    if np.linalg.norm(axis) > 1e-6:  # Avoid division by zero
        axis = axis / np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        mshArrow.rotate(R, center=(0, 0, 0))

        # Translate to the origin position
    mshArrow.translate(origin)

    return mshArrow


class TransformVisualizer:
    def __init__(self, model: Model, scene: Scene, dictResults: dict):
        self.oModel = model
        self.oScene = scene
        self.dictResults = dictResults

    def renderFoundObjects(self):
        geometries = [self.oScene.pcdViz]

        for _id, result in self.dictResults.items():
            pcdModelCopy = o3d.geometry.PointCloud()
            pcdModelCopy.points = self.oModel.pcdModel.points

            ## Create arrow to show picking pose
            pick_position = self.oModel.getPickPosition()
            pick_normal = self.oModel.getPickNormal()

            mshArrow = create_arrow(pick_position, pick_normal)

            transform = result[0]
            fitness = result[1]
            inlier_rmse = result[2]

            # logger.info(f"Result for object {_id}; Fitness: {fitness}, RMSE: {inlier_rmse}")

            # print(transform)

            pcdObjectTransformed = pcdModelCopy.transform(transform)
            geometries.append(pcdObjectTransformed)

            ## Transform the arrow and add to geometries
            mshArrow.transform(transform)
            geometries.append(mshArrow)


        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        for geo in geometries:
            vis.add_geometry(geo)

        vis.poll_events()
        vis.update_renderer()

        # Set camera viewpoint
        # Get the center of the point cloud (using the bounding box's center as an example)
        bbox = self.oScene.pcdROI.get_axis_aligned_bounding_box()
        point_cloud_center = bbox.get_center()  # This is the center of your point cloud
        eye = point_cloud_center

        center = point_cloud_center + np.array([0, 0, 1], dtype=np.float32)

        up = np.array([1, 1, 0], dtype=np.float32)

        ctr = vis.get_view_control()
        # ctr.set_lookat(center)
        ctr.set_front(center - eye)
        ctr.set_up(up)

        ## Capture image
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        image_np = (np.asarray(image)*255).astype(np.uint8)

        cv2.imwrite("render.png", image_np)
        return image_np

    def displayFoundObjects(self):
        pcdViz = copy.deepcopy(self.oScene.pcdViz)

        geometries = [pcdViz]


        for _id, result in self.dictResults.items():
            pcdModel = self.oModel.pcdModel

            pcdModelCopy = o3d.geometry.PointCloud()
            pcdModelCopy.points = self.oModel.pcdModel.points

            ## Create arrow to show picking pose
            pick_position = self.oModel.getPickPosition()
            pick_normal = self.oModel.getPickNormal()

            # logger.debug(f"Pick Position: {pick_position}")
            # logger.debug(f"Pick Normal: {pick_normal}")

            mshArrow = create_arrow(pick_position, pick_normal)

            transform = result[0]
            fitness = result[1]
            inlier_rmse = result[2]

            # logger.info(f"Result for object {_id}; Fitness: {fitness}, RMSE: {inlier_rmse}")

            pcdObjectTransformed = pcdModelCopy.transform(transform)
            geometries.append(pcdObjectTransformed)

            ## Transform the arrow and add to geometries
            mshArrow.transform(transform)
            geometries.append(mshArrow)

        display_point_clouds(geometries, "Found objects", False, True, 100)


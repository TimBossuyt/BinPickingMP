import open3d as o3d
import cv2
import logging
import numpy as np
import copy

from .model import Model
from .scene import Scene
from .utils import display_point_clouds, create_arrow

logger = logging.getLogger("Pose Visualizer")

class TransformVisualizer:
    def __init__(self, model: Model, scene: Scene, dictResults: dict):
        self.oModel = model
        self.oScene = scene
        self.dictResults = dictResults

    def renderFoundObjects(self):
        geometries = [self.oScene.pcdViz]

        # display_point_clouds(geometries, "Scene", False, True, 100)

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
            iou = result[3]

            logger.info(f"Result for object {_id}; Fitness: {fitness}, RMSE: {inlier_rmse}, IoU: {iou}")

            if fitness == 0:
                continue

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

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./Output/render.png", image_np)
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


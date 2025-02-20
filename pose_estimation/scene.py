import cv2
import open3d as o3d
from object_masks import ObjectMasks
from pathlib import Path
from segmentation import ObjectSegmentation
import numpy as np

class Scene:
    """
    Manages scene pointcloud for pose estimation
    """

    def __init__(self, pcd, iWidthImage, iHeightImage):
        ## Save coloured raw pointcloud
        self.pcdRaw = pcd


        ## Create image (BGR) from pointcloud data
        self.arrColours = np.asarray(self.pcdRaw.colors).reshape(iHeightImage, iWidthImage, 3) * 255
        self.arrColours = np.uint8(self.arrColours)

        ## Initialize the object segmentation object
        oSegmentation = ObjectSegmentation(500, 100, 1500, 800)
        self.oMasks = ObjectMasks(self.arrColours, oSegmentation)




if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(Path("2025-02-20_17-17-11.ply"))

    oScene = Scene(pcd, 1920, 1080)

    cv2.imshow("Image from pointcloud", oScene.arrColours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    oScene.oMasks.debugSegmentation()

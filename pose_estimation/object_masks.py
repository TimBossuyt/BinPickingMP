import numpy as np
from matplotlib import pyplot as plt
from segmentation import ObjectSegmentation
import cv2
import logging

## Create object masks (boolean) from segmented image

logger = logging.getLogger("ObjectMasks")

class ObjectMasks:
    def __init__(self, image, oSegmentation : ObjectSegmentation):
        self.image = image
        self.oSegmentation = oSegmentation

        self.segmented_image = self.oSegmentation.getSegmentatedImage(image)

        self.object_ids = []

        self.masks = {}

        ## Create array of object ids
        self.__getObjectIds()

        ## Create the masks dictionary key = id, value = binary mask
        self.__createMaskDict()
        
    def getMasks(self):
        return self.masks

    def visSegmentedObjectsWithMasks(self):
        img_annot = self.image.copy()

        for id, mask in self.masks.items():
            # Create an overlay for the mask
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # Random color for each object

            img_annot[mask==1] = color

        # Show the image with the object masks applied
        plt.imshow(cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB))
        plt.title("Segmented Objects with Masks")
        plt.axis('off')
        plt.show()
            
    def debugSegmentation(self):
        self.oSegmentation.debugVisuals()

    def __createMaskDict(self):
        for id in self.object_ids:
            mask = np.zeros_like(self.segmented_image)
            mask[self.segmented_image == id] = 1
            self.masks[id] = mask

    def __getObjectIds(self):
        ## Distinguishing background from detected objects
        # Check number of pixels that correspond to each object
        # Id with biggest number of pixels = background object

        ids, count = np.unique(self.segmented_image, return_counts=True)

        ## Only take positive values (-1 = noise)
        positive_ids = ids[ids > 0]
        positive_count = count[ids > 0]

        ## Find background ID
        if len(positive_ids) > 0:
            background_id = positive_ids[np.argmax(positive_count)]
            self.object_ids = positive_ids[positive_ids != background_id]

            logger.debug(f"Background ID: {background_id}")
            logger.debug(f"Object IDs: {self.object_ids}")
        else:
            ## TODO: Case no objects were found
            logger.warning("No objects were found")
            pass


if __name__ == "__main__":
    img = cv2.imread("segmentation_example.jpg")

    oSegmentation = ObjectSegmentation(500, 100, 1500, 800)
    oMasks = ObjectMasks(img, oSegmentation)

    oMasks.visSegmentedObjectsWithMasks()


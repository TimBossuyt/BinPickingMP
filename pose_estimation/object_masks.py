import numpy as np
from matplotlib import pyplot as plt
from segmentation import ObjectSegmentation
import cv2
import logging

## Create object masks (boolean) from segmented image

logger = logging.getLogger("ObjectMasks")

class ObjectMasks:
    """
    ObjectMasks class provides functionality for handling and visualizing object masks from a segmented image.

    Attributes:
        image (np.ndarray): Original input image.
        oSegmentation (ObjectSegmentation): ObjectSegmentation instance for segmenting the image.
        segmented_image (np.ndarray): Segmented version of the input image produced by ObjectSegmentation.
        object_ids (list[int]): List of IDs corresponding to detected objects in the segmented image.
        masks (dict[int, np.ndarray]): Dictionary where keys are object IDs and values are binary masks.

    Methods:
        __init__(image, segmentation):
            Initializes the ObjectMasks object with an image and an instance of ObjectSegmentation.
            Creates a segmented version of the image, identifies object IDs, and generates binary masks.

        getMasks() -> dict[int, np.ndarray]:
            Returns a dictionary containing object IDs as keys and their corresponding binary masks as values.

        visSegmentedObjectsWithMasks() -> None:
            Visualizes the segmented objects using their masks by overlaying them on the original image.

        debugSegmentation() -> None:
            Invokes the debug visualization method provided by the ObjectSegmentation instance to facilitate debugging.

        __createMaskDict() -> None:
            Generates the binary masks for each object ID found in the segmented image.
            Populates the `masks` dictionary.

        __getObjectIds() -> None:
            Identifies unique object IDs within the segmented image, excluding background and noise.
            Logs the background ID and object IDs using the logger.
    """
    def __init__(self, image: np.ndarray, segmentation : ObjectSegmentation):
        self.image = image
        self.oSegmentation = segmentation

        self.segmented_image = self.oSegmentation.getSegmentatedImage(image)

        self.object_ids = []

        self.masks = {}

        ## Create array of object ids
        self.__getObjectIds()

        ## Create the masks dictionary key = id, value = binary mask
        self.__createMaskDict()
        
    def getMasks(self) -> dict[int, np.ndarray]:
        """
        Returns a dictionary containing mask data.

        :return: A dictionary where the key is an integer and the value is a numpy ndarray representing mask data.
        """
        return self.masks

    def visSegmentedObjectsWithMasks(self) -> None:
        """
        Generates a visualization of segmented objects with applied masks on the image.

        This function overlays masks corresponding to segmented objects on a copy of
        the original image using randomly generated colors for each object mask.
        It then displays the processed image with the segmented objects as visual overlays.

        :return: None
        """
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
            
    def debugSegmentation(self) -> None:
        """
        Calls the debugVisuals method of the oSegmentation object to provide debugging information related to segmentation.

        :return: None
        """

        self.oSegmentation.debugVisuals()

    def __createMaskDict(self) -> None:
        """
        Creates a dictionary of binary masks for each object ID in the segmented image.

        Each binary mask is a NumPy array of the same shape as the segmented image,
        representing the pixels belonging to a specific object ID. Pixels corresponding
        to the given object ID are set to 1, while others are set to 0. The generated masks
        are stored in the `self.masks` dictionary, with the object ID as the key.

        :return: None
        """
        for _id in self.object_ids:
            mask = np.zeros_like(self.segmented_image)
            mask[self.segmented_image == _id] = 1
            self.masks[_id] = mask

    def __getObjectIds(self) -> None:
        """
        Distinguishes the background from detected objects and identifies object IDs.

        This method analyzes the segmented image to determine unique object IDs based
        on pixel count. The background object is identified as the object with the
        highest pixel count, and other object IDs are extracted after excluding
        the background object. If no objects are found, an appropriate warning is logged.

        :return: None
        """

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


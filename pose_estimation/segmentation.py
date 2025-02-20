import numpy as np
import cv2
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger("Segmentation")

class ObjectSegmentation:
    """
    ObjectSegmentation is a class for performing image segmentation using the Watershed algorithm.
    The class allows segmentation within a predefined region of interest (ROI)
    and includes methods for thresholding, morphological operations, and visualization of the segmentation process.
    """
    def __init__(self, x_min: int, y_min:int, x_max:int, y_max:int):
        logger.debug("Initializing ObjectSegmentation object with ROI: "
                     "x_min=%d, y_min=%d, x_max=%d, y_max=%d", x_min, y_min, x_max, y_max)

        self.cvImage = None

        ## Define ROI parameters
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        ## Thresholding init
        self.sure_fg = None
        self.sure_bg = None
        self.unknown = None
        self.dist_transform = None
        self.markers = None
        self.img_cleaned = None

        ## Final watershed image
        self.watershed_img = None


    def getSegmentatedImage(self, cvImg: np.ndarray) -> np.ndarray:
        """
        :param cvImg: Input image in the form of a numpy ndarray.
        :return: A segmented image as a numpy ndarray with shape (1080, 1920), where each integer value indicates a segmented object.
        """

        logger.debug(f"Input image shape: {cvImg.shape}")
        self.__watershed(cvImg)

        logger.info("Watershed segmentation completed successfully")
        return self.watershed_img

    def __watershed(self, image: np.ndarray) -> None:
        """
        :param image: The input image as a NumPy ndarray, in BGR color space.
        :return: None
        """

        ## Set img attribute
        self.cvImage = image

        ## 1. Convert to gray colour scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ## 2. Distinguish foreground (255) from background (0) using Otsu's method
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        logger.debug(f"Thresholding completed; threshold shape: {thresh.shape}")

        ## 3. Define everything outside ROI as background
        thresh = self.__applyROI(thresh)

        ## 4. Threshold image processing
        self.__processMaskedImage(thresh)

        ## 5. Create labeling markers
        self.__createMarkers()

        self.watershed_img = cv2.watershed(self.cvImage, self.markers)


    def __applyROI(self, img_thresh: np.ndarray) -> np.ndarray:
        """
        :param img_thresh: Input thresholded image as a NumPy array.
        :return: ROI-applied thresholded image as a NumPy array.
        """

        thresh_masked = np.zeros_like(img_thresh)

        thresh_masked[self.y_min:self.y_max, self.x_min:self.x_max] = img_thresh[self.y_min:self.y_max, self.x_min:self.x_max]

        return thresh_masked

    def __createMarkers(self):
        """
        Creates and initializes the markers for the Watershed algorithm.

        This method utilizes the connected components of the foreground object (`sure_fg`)
        to generate markers required for the Watershed segmentation. It increments all
        marker values by 1 to ensure the background, previously marked as 0, is not
        treated as unknown. Regions marked as `unknown` are set to 0 in the markers.

        :return: None. Modifies the instance attribute `self.markers`.
        """

        _, self.markers = cv2.connectedComponents(self.sure_fg)
        ## Don't mark background (0 -> 1) as unknown (0)
        self.markers = self.markers + 1

        ## Mark unknown as 0
        self.markers[self.unknown == 255] = 0

        logger.info("Markers for Watershed algorithm created successfully")

        logger.debug(f"Markers shape: {self.markers.shape}")


    def __processMaskedImage(self, img_masked: np.ndarray) -> None:
        """
        Processes a masked image to clean noise, close holes, and define regions for further analysis
        such as sure background, sure foreground, and unknown regions.

        Involves morphological operations and distance transformation.

        :param img_masked: Input masked image as a numpy ndarray to be processed.
        :return: None
        """

        # TODO: Remove hardcoded kernels

        ## Remove noise using morphological opening (= erode followed by dilute)
        kernel = np.ones((3, 3), np.uint8)
        img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations=3)

        ## Close holes in objects using morphological closing ( = dilute followed by erosion)
        kernel = np.ones((5, 5),np.uint8)
        img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel, iterations=2)

        ## Save masked image after removing noise + closing holes
        self.img_cleaned = img_masked

        logger.debug("Masked image cleaned and saved")

        ## Define sure background
        self.sure_bg = cv2.dilate(img_masked, kernel, iterations=3)

        ## Define sure foreground based on distance transform (euclidian distance to nearest black pixel)
        self.dist_transform = cv2.distanceTransform(img_masked, cv2.DIST_L2, 5)
        _, self.sure_fg = cv2.threshold(self.dist_transform, 0.5*self.dist_transform.max(), 255, 0)

        ## Figure out the unknown region (sure background - sure foreground)
        self.sure_fg = np.uint8(self.sure_fg)
        self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)

    def debugVisuals(self):
        """
        Displays a series of visualizations in a 2x3 grid format showing different stages or aspects of image processing.
        The stages include segmented image, cleaned image, sure background, sure foreground, unknown region,
        and distance transform. Each image is shown in grayscale, with titles corresponding to the stage it represents.

        :return: None
        """

        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
        images = [self.watershed_img, self.img_cleaned, self.sure_bg, self.sure_fg, self.unknown, self.dist_transform]
        titles = ["Segmentated Image", "Cleaned Image", "Sure Background", "Sure Foreground", "Unknown Region",
                  "Distance Transform"]
        for ax, image, title in zip(axes.flat, images, titles):
            ax.imshow(image, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    img = cv2.imread("segmentation_example.jpg")

    oSegmentation = ObjectSegmentation(500, 100, 1500, 800)

    segm_image = oSegmentation.getSegmentatedImage(img)

    ## Debugging visuals
    oSegmentation.debugVisuals()


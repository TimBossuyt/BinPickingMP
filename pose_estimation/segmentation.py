import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import logging
from .settings import SettingsManager

logger = logging.getLogger("Segmentation")


## Disable debug logging messages for matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Set non-GUI backend to avoid the warning
matplotlib.use("Agg")


class ObjectSegmentation:

    """
    ObjectSegmentation is a class for performing image segmentation using the Watershed algorithm.
    The class allows segmentation within a predefined region of interest (ROI)
    and includes methods for thresholding, morphological operations, and visualization of the segmentation process.
    """

    def __init__(self, oSettingsManager: SettingsManager):
        self.oSm = oSettingsManager
        self.__loadSettings()

        self.cvImage = None

        ## Thresholding init
        self.sure_fg = None
        self.sure_bg = None
        self.unknown = None
        self.dist_transform = None
        self.markers = None
        self.img_cleaned = None

        ## Final watershed image
        self.watershed_img = None

    def __loadSettings(self) -> None:
        ## --------------- ROI ---------------
        self.x_min = self.oSm.get("ObjectSegmentation.ROI.xMin")
        self.y_min = self.oSm.get("ObjectSegmentation.ROI.yMin")
        self.x_max = self.oSm.get("ObjectSegmentation.ROI.xMax")
        self.y_max = self.oSm.get("ObjectSegmentation.ROI.yMax")

        ## ---------------- Morphological Operations ------------
        kernel = self.oSm.get("ObjectSegmentation.MorphologicalProcessing.OpeningKernel")
        self.OpeningKernelSize = (kernel[0], kernel[1])
        self.OpeningIterations = self.oSm.get("ObjectSegmentation.MorphologicalProcessing.OpeningIterations")

        kernel = self.oSm.get("ObjectSegmentation.MorphologicalProcessing.ClosingKernel")
        self.ClosingKernelSize = (kernel[0], kernel[1])
        self.ClosingIterations = self.oSm.get("ObjectSegmentation.MorphologicalProcessing.ClosingIterations")

        ## ---------------- Sure Background ------------
        self.sureBgIterations = self.oSm.get("ObjectSegmentation.SureBackground.Iterations")

        ## ---------------- Distance transform ------------
        self.DistTransformMask = self.oSm.get("ObjectSegmentation.DistanceTransform.MaskSize")

        ## ---------------- Sure Foreground ------------
        self.SureFgThreshold = self.oSm.get("ObjectSegmentation.SureForeground.Threshold")

        logger.debug("Settings set correctly")


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

        cv2.imwrite("segm_image.jpg", self.cvImage)

        ## 1. Convert to gray colour scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Gray image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        roi = gray[self.y_min:self.y_max, self.x_min:self.x_max]

        ## 2. Distinguish foreground (255) from background (0) using Otsu's method
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        logger.debug(f"Thresholding completed; threshold shape: {roi_thresh.shape}")

        ## 3. Define everything outside ROI as background
        mask = np.zeros_like(gray)
        mask[self.y_min:self.y_max, self.x_min:self.x_max] = roi_thresh

        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # thresh = self.__applyROI(thresh)

        ## 4. Threshold image processing
        self.__processMaskedImage(mask)

        ## 5. Create labeling markers
        self.__createMarkers()

        self.watershed_img = cv2.watershed(self.cvImage, self.markers)

        # cv2.imshow("Watershed result", self.watershed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def __applyROI(self, img_thresh: np.ndarray) -> np.ndarray:
        """
        :param img_thresh: Input thresholded image as a NumPy array.
        :return: ROI-applied thresholded image as a NumPy array.
        """

        ## Define everything outside of ROI as background
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
        kernel = np.ones(self.OpeningKernelSize, np.uint8)
        img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations=self.OpeningIterations)

        ## Close holes in objects using morphological closing ( = dilute followed by erosion)
        kernel = np.ones(self.ClosingKernelSize,np.uint8)
        img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel, iterations=self.ClosingIterations)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_masked, connectivity=8)

        # Filter out small components
        min_area = 10000  # Change based on expected object size
        for i in range(1, num_labels):  # Ignore background (label 0)
            print(stats[i, cv2.CC_STAT_AREA])
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                img_masked[labels == i] = 0

        cv2.imshow("Mask - cleaned", img_masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ## Save masked image after removing noise + closing holes
        self.img_cleaned = img_masked

        logger.debug("Masked image cleaned and saved")

        ## Define sure background
        self.sure_bg = cv2.dilate(img_masked, kernel, iterations=self.sureBgIterations)

        ## Define sure foreground based on distance transform (euclidian distance to nearest black pixel)
        self.dist_transform = cv2.distanceTransform(img_masked, cv2.DIST_L2, self.DistTransformMask)
        _, self.sure_fg = cv2.threshold(self.dist_transform, self.SureFgThreshold*self.dist_transform.max(), 255, 0)

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

        logger.info("Saving segmentation debug visuals")
        plt.savefig("debug_segmentation_visuals.png")
        plt.close(fig)


if __name__ == '__main__':
    img = cv2.imread("test_input/2025-02-20_19-46-58.jpg")
    sm = SettingsManager("default_settings.json")


    oSegmentation = ObjectSegmentation(sm)

    segm_image = oSegmentation.getSegmentatedImage(img)

    ## Debugging visuals
    oSegmentation.debugVisuals()


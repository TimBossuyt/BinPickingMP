import numpy as np
import cv2
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger("Segmentation")

class ObjectSegmentation:
    def __init__(self, x_min, y_min, x_max, y_max):
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


    def getSegmentatedImage(self, cvImg):
        ## Returns array (1080, 1920) with integer values indicating segmented object
        ## TODO: Figure out what to do with segmentated background

        logger.info("Starting segmentation process")
        logger.debug(f"Input image shape: {cvImg.shape}")
        self.__watershed(cvImg)

        logger.info("Watershed segmentation completed successfully")
        return self.watershed_img

    def __watershed(self, img):
        logger.info("Performing Watershed algorithm")
        ## Set img attribute
        self.cvImage = img

        ## 1. Convert to gray colour scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ## 2. Distinguish foreground (255) from background (0) using Otsu's method
        logger.debug("Applying Otsu's thresholding")
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        logger.debug(f"Thresholding completed; threshold shape: {thresh.shape}")

        ## 3. Define everything outside ROI as background
        logger.debug("Applying ROI to thresholded image")
        thresh = self.__applyROI(thresh)

        ## 4. Threshold image processing
        logger.info("Processing masked image for noise removal and hole closing")
        self.__processMaskedImage(thresh)

        ## 5. Create labeling markers
        logger.info("Creating markers for Watershed algorithm")
        self.__createMarkers()

        self.watershed_img = cv2.watershed(self.cvImage, self.markers)


    def __applyROI(self, img_thresh):
        thresh_masked = np.zeros_like(img_thresh)

        thresh_masked[self.y_min:self.y_max, self.x_min:self.x_max] = img_thresh[self.y_min:self.y_max, self.x_min:self.x_max]

        return thresh_masked

    def __createMarkers(self):
        _, self.markers = cv2.connectedComponents(self.sure_fg)
        ## Don't mark background (0 -> 1) as unknown (0)
        self.markers = self.markers + 1

        ## Mark unknown as 0
        self.markers[self.unknown == 255] = 0

        logger.info("Markers for Watershed algorithm created successfully")

        logger.debug(f"Markers shape: {self.markers.shape}")


    def __processMaskedImage(self, img_masked):
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


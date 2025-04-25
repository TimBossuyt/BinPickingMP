import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from ultralytics import SAM
import logging
from .settings import SettingsManager

logger = logging.getLogger("Segmentation")

## Disable debug logging messages for matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Set non-GUI backend to avoid the warning
matplotlib.use("Agg")

class ObjectSegmentation:
    def __init__(self, oSettingsManager: SettingsManager, model):
        self.oSm = oSettingsManager
        self.__loadSettings()
        self.SamModel = model

    def __loadSettings(self) -> None:
        ## --------------- ROI ---------------
        value = self.oSm.get("ObjectSegmentation.bVisualize")
        self.bVisualize = (value == 1)

        logger.debug("Settings set correctly")

    def getMasksFromImage(self, image):
        ## Thresholding settings
        # h_min, h_max = 0.502, 0.715
        # s_min, s_max = 0.000, 0.365
        # v_min, v_max = 0.100, 0.539

        h_min, h_max = 0.112, 0.177
        s_min, s_max = 0.095, 0.315
        v_min, v_max = 0, 1

        lower = np.array([h_min * 180, s_min * 255, v_min * 255], np.uint8)
        upper = np.array([h_max * 180, s_max * 255, v_max * 255], np.uint8)

        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_all = cv2.inRange(img_hsv, lower, upper)

        if self.bVisualize:
            cv2.imshow("Mask", mask_all)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        ## Filtering
        kernel = np.ones((10, 10), np.uint8)
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel)
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel)

        if self.bVisualize:
            cv2.imshow("Enhanced mask", mask_all)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        ## Find contours, select all above certain threshold and fill
        contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_objects = np.zeros_like(mask_all)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 2000:
                cv2.drawContours(mask_objects, [cnt], -1, 255, -1)

        if self.bVisualize:
            cv2.imshow("Contour filtering", mask_objects)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        mask_objects = cv2.erode(mask_objects, kernel, iterations=2)

        if self.bVisualize:
            cv2.imshow("Eroded", mask_objects)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        ## Apply distance transform
        dist_transform = cv2.distanceTransform(mask_objects, cv2.DIST_L2, 5)

        dist_transform_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_transform_norm = np.uint8(dist_transform_norm)

        if self.bVisualize:
            cv2.imshow("Distance Transform", dist_transform_norm)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        _, thresholded_dist_transform = cv2.threshold(dist_transform_norm, 200, 255, cv2.THRESH_BINARY)

        if self.bVisualize:
            cv2.imshow("Thresholded Distance Transform", thresholded_dist_transform)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Find connected components on the thresholded distance transform to get centroids
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_dist_transform,
                                                                                connectivity=8)
        # Display the centroids on the original image (or thresholded distance transform for visualization)
        image_with_centroids = image.copy()

        # Iterate over each centroid and mark it on the image
        for c in centroids[1:]:  # Skipping the first one since it corresponds to the background
            cx, cy = int(c[0]), int(c[1])  # Convert centroid coordinates to integers
            cv2.circle(image_with_centroids, (cx, cy), 5, (255, 0, 0), -1)  # Draw a red circle at the centroid

        # Show the image with centroids
        if self.bVisualize:
            cv2.imshow("Centroids", image_with_centroids)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        centroids = centroids[1:]
        n_objects = len(centroids)

        if n_objects == 0:
            return []

        print(f"Trying to find {n_objects} objects")

        ## Image enhancement
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        img_lab_clahe = cv2.merge((l_clahe, a, b))

        img_bgr_enhanced = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2BGR)

        if self.bVisualize:
            cv2.imshow("Enhanced Image", img_bgr_enhanced)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        smoothed = cv2.GaussianBlur(img_bgr_enhanced, (19, 19), sigmaX=0)

        if self.bVisualize:
            cv2.imshow("Smoothed image", smoothed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        image_enhanced_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)

        results = self.SamModel(image_enhanced_rgb, points=centroids, max_det=n_objects, labels=np.ones(n_objects),
                        device='cuda', conf=0.5, retina_masks=False)

        masks = results[0].masks.data.cpu().numpy()

        overlay = image.copy()

        print(f"Found {len(masks)} objects using SAM")

        processed_masks = []

        for mask in masks:
            mask = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_bool = mask.astype(bool)

            processed_masks.append(mask_bool)

            color = np.random.randint(100, 255, (3,), dtype=np.uint8)

            # Create a colored mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask_bool] = color

            # Blend it with the original image
            overlay = np.where(mask_bool[..., None], 0.5 * overlay + 0.5 * colored_mask, overlay)

        overlay = cv2.cvtColor(np.uint8(overlay), cv2.COLOR_BGR2RGB)

        # Show the final result
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay.astype(np.uint8))
        plt.axis("off")
        plt.title("Segmented objects")
        # plt.show()
        plt.savefig("SegmentObjects.jpg")

        unique_masks = []
        for i, mask in enumerate(processed_masks):
            is_duplicate = False
            for um in unique_masks:
                if self.compute_iou(mask, um) > 0.8:  # Threshold can be tuned
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_masks.append(mask)


        print(f"Reduced to {len(unique_masks)} unique masks")

        print("Mask sizes")
        for i, mask in enumerate(unique_masks):
            mask_size = np.count_nonzero(mask)

            print(f"Object {i} - {mask_size}")

            if mask_size < 20000:
                unique_masks.pop(i)
                print(f"Object {i} was removed")

        return unique_masks

    def reload_settings(self):
        self.__loadSettings()
        logger.info("Reloaded settings")

    @staticmethod
    def compute_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0

if __name__ == '__main__':
    img = cv2.imread("test_input/50mm.jpg")
    sm = SettingsManager("../settings.json")
    model = SAM("../sam2.1_b.pt")

    oSegmentation = ObjectSegmentation(sm, model)

    segm_image = oSegmentation.getMasksFromImage(img)


    # def getSegmentatedImage(self, cvImg: np.ndarray) -> np.ndarray:
    #     """
    #     :param cvImg: Input image in the form of a numpy ndarray.
    #     :return: A segmented image as a numpy ndarray with shape (1080, 1920), where each integer value indicates a segmented object.
    #     """
    #
    #     logger.debug(f"Input image shape: {cvImg.shape}")
    #     self.__watershed(cvImg)
    #
    #     logger.info("Watershed segmentation completed successfully")
    #     return self.watershed_img

    # def __watershed(self, image: np.ndarray) -> None:
    #     """
    #     :param image: The input image as a NumPy ndarray, in BGR color space.
    #     :return: None
    #     """
    #
    #     ## Set img attribute
    #     self.cvImage = image
    #
    #     cv2.imwrite("segm_image.jpg", self.cvImage)
    #
    #     ## 1. Convert to gray colour scale
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     cv2.imshow("Gray image", gray)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     roi = gray[self.y_min:self.y_max, self.x_min:self.x_max]
    #
    #     ## 2. Distinguish foreground (255) from background (0) using Otsu's method
    #     _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    #     logger.debug(f"Thresholding completed; threshold shape: {roi_thresh.shape}")
    #
    #     ## 3. Define everything outside ROI as background
    #     mask = np.zeros_like(gray)
    #     mask[self.y_min:self.y_max, self.x_min:self.x_max] = roi_thresh
    #
    #     cv2.imshow("Mask", mask)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     # thresh = self.__applyROI(thresh)
    #
    #     ## 4. Threshold image processing
    #     self.__processMaskedImage(mask)
    #
    #     ## 5. Create labeling markers
    #     self.__createMarkers()
    #
    #     self.watershed_img = cv2.watershed(self.cvImage, self.markers)

        # cv2.imshow("Watershed result", self.watershed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


#     def __applyROI(self, img_thresh: np.ndarray) -> np.ndarray:
#         """
#         :param img_thresh: Input thresholded image as a NumPy array.
#         :return: ROI-applied thresholded image as a NumPy array.
#         """
#
#         ## Define everything outside of ROI as background
#         thresh_masked = np.zeros_like(img_thresh)
#
#         thresh_masked[self.y_min:self.y_max, self.x_min:self.x_max] = img_thresh[self.y_min:self.y_max, self.x_min:self.x_max]
#
#         return thresh_masked
#
#     def __createMarkers(self):
#         """
#         Creates and initializes the markers for the Watershed algorithm.
#
#         This method utilizes the connected components of the foreground object (`sure_fg`)
#         to generate markers required for the Watershed segmentation. It increments all
#         marker values by 1 to ensure the background, previously marked as 0, is not
#         treated as unknown. Regions marked as `unknown` are set to 0 in the markers.
#
#         :return: None. Modifies the instance attribute `self.markers`.
#         """
#
#         _, self.markers = cv2.connectedComponents(self.sure_fg)
#         ## Don't mark background (0 -> 1) as unknown (0)
#         self.markers = self.markers + 1
#
#         ## Mark unknown as 0
#         self.markers[self.unknown == 255] = 0
#
#         logger.info("Markers for Watershed algorithm created successfully")
#
#         logger.debug(f"Markers shape: {self.markers.shape}")
#
#
#     def __processMaskedImage(self, img_masked: np.ndarray) -> None:
#         """
#         Processes a masked image to clean noise, close holes, and define regions for further analysis
#         such as sure background, sure foreground, and unknown regions.
#
#         Involves morphological operations and distance transformation.
#
#         :param img_masked: Input masked image as a numpy ndarray to be processed.
#         :return: None
#         """
#
#         # TODO: Remove hardcoded kernels
#
#         ## Remove noise using morphological opening (= erode followed by dilute)
#         kernel = np.ones(self.OpeningKernelSize, np.uint8)
#         img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations=self.OpeningIterations)
#
#         ## Close holes in objects using morphological closing ( = dilute followed by erosion)
#         kernel = np.ones(self.ClosingKernelSize,np.uint8)
#         img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel, iterations=self.ClosingIterations)
#
#         # Find connected components
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_masked, connectivity=8)
#
#         # Filter out small components
#         min_area = 10000  # Change based on expected object size
#         for i in range(1, num_labels):  # Ignore background (label 0)
#             print(stats[i, cv2.CC_STAT_AREA])
#             if stats[i, cv2.CC_STAT_AREA] < min_area:
#                 img_masked[labels == i] = 0
#
#         cv2.imshow("Mask - cleaned", img_masked)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         ## Save masked image after removing noise + closing holes
#         self.img_cleaned = img_masked
#
#         logger.debug("Masked image cleaned and saved")
#
#         ## Define sure background
#         self.sure_bg = cv2.dilate(img_masked, kernel, iterations=self.sureBgIterations)
#
#         ## Define sure foreground based on distance transform (euclidian distance to nearest black pixel)
#         self.dist_transform = cv2.distanceTransform(img_masked, cv2.DIST_L2, self.DistTransformMask)
#         _, self.sure_fg = cv2.threshold(self.dist_transform, self.SureFgThreshold*self.dist_transform.max(), 255, 0)
#
#         ## Figure out the unknown region (sure background - sure foreground)
#         self.sure_fg = np.uint8(self.sure_fg)
#         self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)
#
#     def debugVisuals(self):
#         """
#         Displays a series of visualizations in a 2x3 grid format showing different stages or aspects of image processing.
#         The stages include segmented image, cleaned image, sure background, sure foreground, unknown region,
#         and distance transform. Each image is shown in grayscale, with titles corresponding to the stage it represents.
#
#         :return: None
#         """
#
#         fig, axes = plt.subplots(2, 3, figsize=(8, 6))
#         images = [self.watershed_img, self.img_cleaned, self.sure_bg, self.sure_fg, self.unknown, self.dist_transform]
#         titles = ["Segmentated Image", "Cleaned Image", "Sure Background", "Sure Foreground", "Unknown Region",
#                   "Distance Transform"]
#         for ax, image, title in zip(axes.flat, images, titles):
#             ax.imshow(image, cmap='gray')
#             ax.set_title(title)
#             ax.axis('off')
#         plt.tight_layout()
#
#         logger.info("Saving segmentation debug visuals")
#         plt.savefig("debug_segmentation_visuals.png")
#         plt.close(fig)
#
#
# if __name__ == '__main__':
#     img = cv2.imread("test_input/2025-02-20_19-46-58.jpg")
#     sm = SettingsManager("default_settings.json")
#
#
#     oSegmentation = ObjectSegmentation(sm)
#
#     segm_image = oSegmentation.getSegmentatedImage(img)
#
#     ## Debugging visuals
#     oSegmentation.debugVisuals()


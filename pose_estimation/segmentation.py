import logging

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from .settings import SettingsManager

logger = logging.getLogger("Segmentation")

## Disable debug logging messages for matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Set non-GUI backend to avoid the warning
matplotlib.use("Agg")

class ObjectSegmentation:
    def __init__(self, oSettingsManager: SettingsManager, model):
        self.oSm = oSettingsManager
        self._loadSettings()
        self.SamModel = model

    def _loadSettings(self) -> None:
        ## --------------- ROI ---------------
        value = self.oSm.get("ObjectSegmentation.bVisualize")
        self.bVisualize = (value == 1)

        ## --------------- Color thresholds ---------------
        h_min = self.oSm.get("ObjectSegmentation.HsvThresholding.H.min")
        h_max = self.oSm.get("ObjectSegmentation.HsvThresholding.H.max")
        s_min = self.oSm.get("ObjectSegmentation.HsvThresholding.S.min")
        s_max = self.oSm.get("ObjectSegmentation.HsvThresholding.S.max")
        v_min = self.oSm.get("ObjectSegmentation.HsvThresholding.V.min")
        v_max = self.oSm.get("ObjectSegmentation.HsvThresholding.V.max")

        self.lower = np.array([h_min * 180, s_min * 255, v_min * 255], np.uint8)
        self.upper = np.array([h_max * 180, s_max * 255, v_max * 255], np.uint8)

        ## --------------- SAM ---------------
        self.fSAM_conf = self.oSm.get("ObjectSegmentation.SAM.confidence")
        self.bSAM_retina = (self.oSm.get("ObjectSegmentation.SAM.retinaMasks") == 1)
        self.sSAM_device = self.oSm.get("ObjectSegmentation.SAM.device")

        self.iSizeThresh = self.oSm.get("ObjectSegmentation.SizeThreshold")

        logger.debug("Settings set correctly")

    def getMasksFromImage(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_all = cv2.inRange(img_hsv, self.lower, self.upper)

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

        mask_objects = cv2.erode(mask_objects, kernel, iterations=6)

        if self.bVisualize:
            cv2.imshow("Eroded", mask_objects)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Find connected components on the thresholded distance transform to get centroids
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_objects,
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

        try:
            results = self.SamModel(image_enhanced_rgb, points=centroids, max_det=n_objects, labels=np.ones(n_objects),
                                    device=self.sSAM_device, conf=self.fSAM_conf, retina_masks=self.bSAM_retina)
        except Exception as e:
            raise e

        ## Copy results to cpu as numpy array
        masks = results[0].masks.data.cpu().numpy()

        overlay = image.copy()

        logger.info(f"Found {len(masks)} objects using SAM")

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
        plt.savefig("./Output/SegmentObjects.jpg")
        plt.close()

        unique_masks = []
        for i, mask in enumerate(processed_masks):
            is_duplicate = False
            for um in unique_masks:
                if self.compute_iou(mask, um) > 0.7:  # Threshold can be tuned
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_masks.append(mask)

        logger.info(f"Reduced to {len(unique_masks)} unique masks")

        logger.debug("---------- Mask sizes ----------")
        for i, mask in enumerate(unique_masks):
            mask_size = np.count_nonzero(mask)

            logger.debug(f"Object {i} - {mask_size}")

            if mask_size < self.iSizeThresh:
                unique_masks.pop(i)
                logger.debug(f"Object {i} was removed")

        logger.debug("--------------------------------")

        return unique_masks

    def reload_settings(self):
        self._loadSettings()
        logger.info("Reloaded settings")

    @staticmethod
    def compute_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0

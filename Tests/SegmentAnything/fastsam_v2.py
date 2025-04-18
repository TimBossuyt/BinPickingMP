from ultralytics import FastSAM
from ultralytics import SAM
import cv2
import numpy as np
import matplotlib.pyplot as plt

# model = FastSAM('FastSAM-s.pt')
model = SAM('sam2.1_b.pt')

## ---------- PARAMETERS ----------
x1, y1 = 800, 180
x2, y2 = 1300, 800

## Blue range
lower_blue = np.array([210/360*179, 0, 0])
upper_blue = np.array([270/360*179, 255, 255])


## Object range
lower = np.array([0, 0, 0], np.uint8)
upper = np.array([180, 0.65*255, 0.65*255], np.uint8)

print(lower)
print(upper)

## --------------------------------
image_src = "2025-04-18_10-25-16.jpg"
image = cv2.imread(image_src)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_rgb_roi = image_rgb[y1:y2, x1:x2]
img_hsv_roi = cv2.cvtColor(img_rgb_roi, cv2.COLOR_RGB2HSV)

mask_blue_roi = cv2.inRange(img_hsv_roi, lower_blue, upper_blue)

# mask_objects_roi = cv2.bitwise_not(mask_blue_roi)
mask_objects_roi = cv2.inRange(img_hsv_roi, lower, upper)

mask_full = np.zeros(image.shape[:2], np.uint8)
mask_full[y1:y2, x1:x2] = mask_objects_roi

print(np.unique(mask_full))

cv2.imshow("Mask", mask_full)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Apply filtering
kernel = np.ones((15, 15), np.uint8)
mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)
# mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, (9, 9), iterations=5)

cv2.imshow("Enhanced mask", mask_full)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Remove all small blobs
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_full, connectivity=8)

min_area = 500  # Minimum area threshold for the blobs
mask_cleaned = np.zeros_like(mask_full)

for i in range(1, num_labels):  # Start from 1 to ignore the background
    if stats[i, cv2.CC_STAT_AREA] >= min_area:  # Only keep large blobs
        mask_cleaned[labels == i] = 255


# Show the result after removing small blobs
cv2.imshow("Cleaned Mask", mask_cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply the distance transform
dist_transform = cv2.distanceTransform(mask_cleaned, cv2.DIST_L2, 5)

# Normalize the result to display it
dist_transform_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
dist_transform_norm = np.uint8(dist_transform_norm)

# Show the distance transform result
cv2.imshow("Distance Transform", dist_transform_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply threshold for values above 150
_, thresholded_dist_transform = cv2.threshold(dist_transform_norm, 150, 255, cv2.THRESH_BINARY)

# Show the thresholded distance transform
cv2.imshow("Thresholded Distance Transform", thresholded_dist_transform)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find connected components on the thresholded distance transform to get centroids
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_dist_transform, connectivity=8)

# Display the centroids on the original image (or thresholded distance transform for visualization)
image_with_centroids = image.copy()

# Iterate over each centroid and mark it on the image
for c in centroids[1:]:  # Skipping the first one since it corresponds to the background
    cx, cy = int(c[0]), int(c[1])  # Convert centroid coordinates to integers
    cv2.circle(image_with_centroids, (cx, cy), 5, (255, 0, 0), -1)  # Draw a red circle at the centroid

# Show the image with centroids
cv2.imshow("Centroids", image_with_centroids)
cv2.waitKey(0)
cv2.destroyAllWindows()

# bounding_boxes = []
#
# for i in range(1, num_labels):  # Skip background
#     x, y, w, h, area = stats[i]
#     bounding_boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2] format
#
# image_boxes = image.copy()
#
# # Draw rectangles
# for bbox in bounding_boxes:
#     x1, y1, x2, y2 = bbox
#     cv2.rectangle(image_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# # Display the image
# cv2.imshow("Image with Bounding Boxes", image_boxes)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## ---------------- SAM Segmentation ----------------
centroids = centroids[1:]
n_objects = len(centroids)

print(f"Trying to find {n_objects} objects")


## Histogram equalization
# Get the ROI as before
roi = image[y1:y2, x1:x2]
mask_roi = mask_objects_roi  # From your earlier code

# --- Convert to LAB color space for better contrast control ---
lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_roi)

# --- Apply CLAHE to L channel ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# --- Merge and convert back to BGR ---
lab_clahe = cv2.merge((l_clahe, a, b))
roi_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# --- Replace only masked region in the original ROI ---
enhanced_roi = roi.copy()
enhanced_roi[mask_roi > 0] = roi_clahe[mask_roi > 0]

# Put the enhanced ROI back into the original image
image_enhanced = image.copy()
image_enhanced[y1:y2, x1:x2] = enhanced_roi

# --- Display the result ---
cv2.imshow("Enhanced Image", image_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

smoothed = cv2.GaussianBlur(image_enhanced, (13, 13), sigmaX=0)
cv2.imshow("Smoothed image", smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_enhanced_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)


results = model(image_enhanced_rgb, points=centroids, max_det=n_objects, labels=np.ones(n_objects), device='cuda', conf=0.2, retina_masks=True)
                # bboxes=bounding_boxes)

masks = results[0].masks.data.cpu().numpy()

overlay = image.copy()

print(f"Found {len(masks)} objects using SAM")

for mask in masks:
    mask = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_bool = mask.astype(bool)

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
plt.title("Segmented objects with prompt")
plt.show()
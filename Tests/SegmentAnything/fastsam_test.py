from ultralytics import FastSAM
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = FastSAM("FastSAM-s.pt")

image_src = "img_box.jpg"
image = cv2.imread(image_src)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_crop = image_rgb[180:800, 800:1300]
hsv = cv2.cvtColor(img_crop, cv2.COLOR_RGB2HSV)

plt.figure(figsize=(10, 10))
plt.imshow(img_crop.astype(np.uint8))
plt.axis("off")
plt.show()

# Define blue color range in HSV
lower_blue = np.array([210/360*179, 20, 20])
upper_blue = np.array([270/360*179, 255, 255])


# Create mask for blue regions
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Invert the mask to get non-blue regions (likely your parts)
non_blue_mask = cv2.bitwise_not(blue_mask)

cv2.imshow("Mask", non_blue_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_blue_mask, connectivity=8)

# Filter out small regions (e.g., noise)
min_area = 500
valid_centroids = [tuple(map(int, c)) for i, c in enumerate(centroids) if stats[i, cv2.CC_STAT_AREA] > min_area and i != 0]

# Visualize detected centroids (optional)
debug_img = img_crop.copy()
for cx, cy in valid_centroids:
    cv2.circle(debug_img, (cx, cy), 6, (255, 0, 0), -1)

plt.figure(figsize=(10, 10))
plt.imshow(debug_img)
plt.title("Detected component centers (Otsu threshold)")
plt.axis("off")
plt.show()

# Run inference with texts prompt
results = model(img_crop, points=valid_centroids, device='cuda', max_det=2, retina_masks=True)

print(results)

# Access the masks
masks = results[0].masks.data.cpu().numpy()  # shape: [num_instances, height, width]

# Overlay each mask on the image (resize masks first!)
overlay = img_crop.copy()
original_h, original_w = img_crop.shape[:2]

for mask in masks:
    # Resize mask to original image size
    mask_resized = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_resized.astype(bool)

    # Generate random color for mask
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)

    # Create a colored mask
    colored_mask = np.zeros_like(img_crop, dtype=np.uint8)
    colored_mask[mask_bool] = color

    # Blend it with the original image
    overlay = np.where(mask_bool[..., None], 0.5 * overlay + 0.5 * colored_mask, overlay)

# Show the final result
plt.figure(figsize=(10, 10))
plt.imshow(overlay.astype(np.uint8))
plt.axis("off")
plt.title("Segmented objects with prompt")
plt.show()
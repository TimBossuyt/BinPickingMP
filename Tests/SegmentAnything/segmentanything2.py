import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

image = cv2.imread('img_box.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the absolute path of your Python script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to the config and checkpoint
model_config = os.path.join(base_dir, "sam2.1_hiera_l.yaml")
model_checkpoint = os.path.join(base_dir, "sam2.1_hiera_large.pt")

start_time = time.time()
sam2_model = build_sam2(model_config, model_checkpoint, device=device)
print(f"Model loading time: {time.time() - start_time:.2f} seconds")

start_time = time.time()
mask_generator = SAM2AutomaticMaskGenerator(sam2_model,
                                            points_per_side=16,
                                            points_per_batch=64,
                                            pred_iou_thresh=0.95,
                                            stability_score_thresh=0.7,
                                            crop_n_layers=0)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,  # Your trained SAM2 model
    points_per_side=32,  # Lower number of points to speed up
    points_per_batch=128,  # Higher batch size for faster processing
    pred_iou_thresh=0.85,  # Higher threshold for higher quality masks
    stability_score_thresh=0.8,  # Accept slightly less stable masks
    mask_threshold=0.7,  # Filter out weak masks
    box_nms_thresh=0.8,  # Filter out duplicate masks
    crop_n_layers=0,  # No need for extra crop layers
    crop_nms_thresh=0.7,  # Not used as crop_n_layers=0
    crop_overlap_ratio=1.0,  # Not used as crop_n_layers=0
    crop_n_points_downscale_factor=1,  # Not used
    point_grids=None,  # Let the class handle point sampling
    min_mask_region_area=0,  # No need for postprocessing
    output_mode='binary_mask',  # Fast output format
    use_m2m=False,  # Skip refinement for speed
    multimask_output=False,  # Only need one mask per object
)

print(f"Mask generator loading time: {time.time() - start_time:.2f} seconds")

## Define ROI
x1 = 800
y1 = 100
x2 = 1500
y2 = 940
image_roi = image[y1:y2, x1:x2]


start_time = time.time()
masks = mask_generator.generate(image_roi)
print(f"Mask generation time: {time.time() - start_time:.2f} seconds")


print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20, 20))
plt.imshow(image_roi)
show_anns(masks)
plt.axis('off')
plt.show()


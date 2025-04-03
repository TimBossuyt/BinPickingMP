import cv2
from pose_estimation import ObjectSegmentation, SettingsManager

sPathImage = "./test_input/2025-03-14_15-51-53.jpg"

img = cv2.imread(sPathImage)
sm = SettingsManager("../pose_settings.json")

oSegmentation = ObjectSegmentation(sm)

segm_image = oSegmentation.getSegmentatedImage(img)

oSegmentation.debugVisuals()

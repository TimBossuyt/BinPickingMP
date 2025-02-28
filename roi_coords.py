import matplotlib.pyplot as plt
import cv2

# Load the image
image_path = "segm_image.jpg"  # Change this to your image file
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors

# Display the image
plt.imshow(image)
plt.axis("on")  # Keep axes visible to see coordinates
plt.show()
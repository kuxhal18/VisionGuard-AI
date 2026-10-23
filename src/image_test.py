
from PIL import Image, ImageDraw
import numpy as np
import cv2

# Create a small sample image
image = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(image)

# Draw a sample object and a simulated defect
draw.rectangle((60, 60, 195, 195), outline="black", width=5)
draw.ellipse((120, 110, 145, 135), fill="red")

# Convert the image into an array for computer vision
image_array = np.array(image)

# Convert RGB to grayscale using OpenCV
gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

# Save the test image
image.save("sample_image.png")

print("Image created successfully!")
print("Image dimensions:", image.size)
print("Array shape:", image_array.shape)
print("Grayscale shape:", gray_image.shape)
print("Saved as sample_image.png")

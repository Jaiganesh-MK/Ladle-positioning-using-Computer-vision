import cv2
import numpy as np

# Load the input image
img = cv2.imread('1.jpg')

# Create a black background image of size 1000x1000
background = np.zeros((1000, 1000, 3), dtype=np.uint8)

# Calculate the coordinates to paste the image onto the background
x = (background.shape[1] - img.shape[1]) // 2
y = (background.shape[0] - img.shape[0]) // 2

# Add black borders to the input image using cv2.copyMakeBorder()
img_with_border = cv2.copyMakeBorder(img, y, y, x, x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Paste the input image onto the background at the calculated coordinates
background[y:y+img.shape[0], x:x+img.shape[1]] = img_with_border

# Display the result
cv2.imshow('Result', background)
cv2.waitKey(0)
cv2.destroyAllWindows()

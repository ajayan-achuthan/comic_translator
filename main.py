import cv2
import numpy as np

# with ideas from:
# http://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
# http://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
print(cv2.__file__)

# Read image
im_in = cv2.imread("gIEXY.png", cv2.IMREAD_GRAYSCALE);

# Threshold.
# Set values equal to or above 200 to 0.
# Set values below 200 to 255.

th, im_th = cv2.threshold(im_in, 200, 255, cv2.THRESH_BINARY_INV);

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from points inside baloons
cv2.floodFill(im_floodfill, mask, (80,400), 128);
cv2.floodFill(im_floodfill, mask, (610,90), 128);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground
im_out = im_th | im_floodfill_inv

# Create binary image from segments with holes
th, im_th2 = cv2.threshold(im_out, 130, 255, cv2.THRESH_BINARY)

# Create contours to fill holes
im_th3 = cv2.bitwise_not(im_th2)
contour,hier = cv2.findContours(im_th3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    cv2.drawContours(im_th3,[cnt],0,255,-1)

segm = cv2.bitwise_not(im_th3)


# Display image
cv2.imshow("Original", im_in)
cv2.imshow("Segmented", segm)
cv2.waitKey(0)
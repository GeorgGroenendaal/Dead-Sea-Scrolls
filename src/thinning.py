


# https://theailearner.com/tag/thinning-opencv/

#######################################################
#img1 = img.copy()
# Structuring Element
import cv2
import numpy as np

img = cv2.imread('C:/Users/mw0121921/Documents/Dead-Sea-Scrolls/data/out/segments/P344-Fg001-R-C01-R01-binarized/segment_820.png',cv2.IMREAD_GRAYSCALE)
img1 = img.copy()
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# # Create an empty output image to hold values
thin = np.zeros(img.shape,dtype='uint8')

# Loop until erosion leads to an empty set
while (cv2.countNonZero(img1)!=0):
     # Erosion
     erode = cv2.erode(img1,kernel)
    # Opening on eroded image
     opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
    # Subtract these two
     subset = erode - opening
     # Union of all previous sets
     thin = cv2.bitwise_or(subset,thin)
     # Set the eroded image for next iteration
     img1 = erode.copy()
    
cv2.imshow('original',img)
cv2.imshow('thinned',thin)
cv2.imwrite("img_out.jpg", thin)
cv2.waitKey(0)

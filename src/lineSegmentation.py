
from pickletools import uint8
from cv2 import CV_32S
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#read input image and invert the pixels: results in white text on black background
#C:/Users/mw0121921/Downloads/image-data/image-data/P583-Fg006-R-C01-R01-binarized.jpg
img = cv2.imread("img_out.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
plt.imshow(img)
plt.show()
img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT) 
img = cv2.bitwise_not(img)

# count pixels per row
hist = cv2.reduce(img, 1, cv2.REDUCE_AVG).reshape(-1)

print(hist)
#mark white pixels in image and bounding boxes
counter = 0
imageCount = 0
#roi = np.zeros(img.shape)
for x in range(0,(img.shape[0]-1)):
    if hist[x] > 100:
        counter += 1
    else:
        if counter > 0:
            roi = img[(x - counter):x,0:(img.shape[1] - 1)]
            cv2.imwrite(str(imageCount)+".jpg", roi)
            imageCount += 1
        counter = 0
                   
# counter = 0 
# image = str(counter) +".jpg" 
# maxColumns = 4
# maxRows = int(imageCount/maxColumns) + 1              
# f, axarr = plt.subplots(maxColumns,maxRows)
# col = 0
# row = 0
# while counter < imageCount:
#     axarr[col,row].imshow(cv2.imread(image))
#     counter += 1
#     image = str(counter) +".jpg" 
#     row += 1
#     if row > (maxRows - 1):
#         row = 0
#     if (counter % maxColumns) == 0:
#         col += 1
# plt.show()






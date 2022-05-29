from math import ceil, floor
import pathlib
from pickle import FALSE
from cv2 import minMaxLoc
import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.signal import find_peaks
import imutils

from src.utils.logger import logger
from src.utils.images import get_name
import matplotlib.pyplot as plt

# from cv2 import IMREAD_GRAYSCALE


class CharacterSegmenter:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def segment_characters(self, image_path: str) -> None:
        name = get_name(image_path)
        image = cv2.imread(image_path)


        # cv2.imshow("Input", image)
        # convert the mean shift image to grayscale, then apply
        # Otsu's thresholding
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
            1
        ]
        
        # cv2.imshow("Thresh", thresh)
        
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        distance = ndimage.distance_transform_edt(thresh)
        #distance = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
        #min, max, min_pos, max_pos = cv2.minMaxLoc(distance)

        
        localMax = peak_local_max(
            distance, indices=False, min_distance=26, labels=thresh,
        )
        print((int(np.max(distance)* 2 )))
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance, markers,  mask=thresh, watershed_line=False)
        logger.info(
            "[INFO] {} unique segments found".format(len(np.unique(labels)) - 1)
        )

        plt.plot(distance)
        plt.show()
        

        # loop over the unique labels returned by the Watershed
        # algorithm
        for i, label in enumerate(np.unique(labels)):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue

            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            mask[labels == label] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)

            circle_mask = np.zeros(gray_image.shape, dtype="uint8")

            x_min, x_max = int(x - r), int(x + r)
            y_min, y_max = int(y - r), int(y + r)
            cv2.circle(circle_mask, (int(x), int(y)), int(r), 255, -1)
            masked = cv2.bitwise_and(gray_image, gray_image, mask=circle_mask) 

            result = masked[ y_min:y_max, x_min:x_max] 
            
            if result.any():
                path = f"data/out/segments/characters/{name}_character_{i}.png"
                parsed_path = pathlib.Path(path)
                parsed_path.parents[0].mkdir(parents=True, exist_ok=True)

                cv2.imwrite(path, result)
                cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # otherwise, allocate memory for the label region and draw
                # it on the mask
           

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey()
           

        if self.debug:
            path = f"data/intermediate/charactersegments/{name}_character.png"
            parsed_path = pathlib.Path(path)
            parsed_path.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(path, image)
        

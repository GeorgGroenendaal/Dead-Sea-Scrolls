import pathlib

import cv2
import imutils
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from src.utils.images import get_name
from src.utils.logger import logger


class CharacterSegmenter:
    def __init__(self, min_distance: int = 40, debug: bool = False):
        self.debug = debug
        self.min_distance = min_distance

    def segment_characters(self, image_path: str) -> None:
        name = get_name(image_path)
        image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
            1
        ]

        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        distance = ndimage.distance_transform_edt(thresh)

        localMax = peak_local_max(
            distance,
            indices=False,
            min_distance=26,
            labels=thresh,
        )

        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance, markers, mask=thresh, watershed_line=False)
        logger.info(
            "[INFO] {} unique segments found".format(len(np.unique(labels)) - 1)
        )

        for i, label in enumerate(np.unique(labels)):
            # if the label is zero, we are examining the 'background'
            if label == 0:
                continue

            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            mask[labels == label] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)

            if self.debug:
                cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)

            circle_mask = np.zeros(gray_image.shape, dtype="uint8")

            # TODO: perform actual circle mask
            x_min, x_max = int(x - r), int(x + r)
            y_min, y_max = int(y - r), int(y + r)
            cv2.circle(circle_mask, (int(x), int(y)), int(r), 255, -1)
            masked = cv2.bitwise_and(gray_image, gray_image, mask=circle_mask)

            result = masked[y_min:y_max, x_min:x_max]

            if result.any() and self.debug:
                path = f"data/out/segments/characters/{name}_character_{i}.png"
                parsed_path = pathlib.Path(path)
                parsed_path.parents[0].mkdir(parents=True, exist_ok=True)

                cv2.imwrite(path, result)
                cv2.putText(
                    image,
                    "#{}".format(label),
                    (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

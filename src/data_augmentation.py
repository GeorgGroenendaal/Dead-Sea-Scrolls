import os

import random
import sys
from typing import List, Tuple
import numpy as np

import cv2

DEFAULT_INPUT_DIRECTORY: str = "data/unpacked/charaters/Alef"
AUGMENTED_DIRECTORY_SUFFIX: str = "_augmented"

EROSION_SIZE = 1
DILATION_SIZE = 1
MAX_ELEM = 2


def readfile() -> None:
    image = cv2.imread(
        "data/unpacked/charaters/Alef/navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm"
    )
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print("test")
    pass


if __name__ == "__main__":
    assert os.path.exists(DEFAULT_INPUT_DIRECTORY)
    # input_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_DIRECTORY
    # print(input_dir)

# readfile()


# erode the image
def erosion(image: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    elements = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * EROSION_SIZE + 1, 2 * EROSION_SIZE + 1),
        (EROSION_SIZE, EROSION_SIZE),
    )

    return cv2.erode(image, elements, iterations=iterations)


# dilate the image
def dilation(image: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    elements = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1),
        (DILATION_SIZE, DILATION_SIZE),
    )

    return cv2.dilate(image, elements, iterations=iterations)

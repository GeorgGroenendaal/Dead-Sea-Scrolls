import pathlib

import cv2
import imutils
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from src.utils.images import get_name, get_parent_name
from src.utils.logger import logger
from src.utils.paths import CHARACTER_SEGMENT_PATH, DEBUG_CHARACTER_SEGMENT_PATH


class CharacterSegmenter:
    def __init__(self, min_distance: int = 40, debug: bool = False):
        self.debug = debug
        self.min_distance = min_distance

    def segment_characters(self, image_path: str) -> None:
        name = get_name(image_path)
        parent_name = get_parent_name(image_path)
        image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        distance = ndimage.distance_transform_edt(thresholded_image)  # distance
        local_max = peak_local_max(
            distance,
            indices=False,
            min_distance=self.min_distance,
            labels=thresholded_image,
        )

        markers, _ = ndimage.label(local_max, structure=np.ones((3, 3)))
        labels = watershed(
            -distance, markers, mask=thresholded_image, watershed_line=False
        )
        logger.info(f"[INFO] {len(np.unique(labels)) - 1} unique segments found")

        for i, label in enumerate(np.unique(labels)):
            if label == 0:  # first label is background
                continue

            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            mask[labels == label] = 255
            contours = cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = imutils.grab_contours(contours)
            max_countours = max(contours, key=cv2.contourArea)
            ((circle_x, circle_y), radius) = cv2.minEnclosingCircle(max_countours)
            circle_mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.circle(
                circle_mask, (int(circle_x), int(circle_y)), int(radius), 255, -1
            )  # in place operation, add circle to mask

            circle_x_min = max(round(circle_x - radius), 0)
            circle_x_max = round(circle_x + radius)
            circle_y_min = max(round(circle_y - radius), 0)
            circle_y_max = round(circle_y + radius)
            result = gray_image * circle_mask * 255
            cropped_result = result[
                circle_y_min:circle_y_max, circle_x_min:circle_x_max
            ]

            out_path = pathlib.Path(
                f"{CHARACTER_SEGMENT_PATH}/{parent_name}/{name}/character_{i}.png"
            )
            out_path.parents[0].mkdir(parents=True, exist_ok=True)

            if cropped_result.any():
                cv2.imwrite(str(out_path), cropped_result)
            else:
                logger.warning(
                    "Cropped result small, skipping, dimensions"
                    f" x {circle_x} y {circle_y} r {circle_y}"
                )

            if result.any() and self.debug:
                cv2.circle(
                    image, (int(circle_x), int(circle_y)), int(radius), (0, 255, 0), 2
                )
                cv2.putText(
                    image,
                    f"#{label}",
                    (int(circle_x) - 10, int(circle_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        if self.debug:
            debug_path = pathlib.Path(f"{DEBUG_CHARACTER_SEGMENT_PATH}/{name}.png")
            debug_path.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), image)

import hashlib
from glob import glob
from typing import List, Set, Tuple

import cv2
from cv2 import Mat
import numpy as np

from src.utils.images import get_name, get_parent_name
from src.utils.logger import logger
from src.utils.paths import CHARACTER_TRAIN_AUGMENTED_PATH, CHARACTER_TRAIN_PATH
import pathlib


def _load_characters(path: str, extension: str = "pgm") -> List[Tuple[str, str, Mat]]:
    paths = _deduplicate_paths(glob(path + f"/**/*.{extension}"))
    logger.info(f"Got {len(paths)} files to load")

    result: List[Tuple[str, str, str]] = []

    for character_path in paths:
        character_name = get_parent_name(character_path)
        file_name = get_name(character_path)
        image = cv2.imread(character_path, cv2.IMREAD_GRAYSCALE)
        result.append((character_name, file_name, image))

    return result


def augment(resize_size: int) -> None:
    # perform augmentation with random variables
    # store in new  folder
    characters = _load_characters(CHARACTER_TRAIN_PATH)

    for character_name, file_name, image in characters:
        image = 255 - image
        image = cv2.resize(image, (resize_size, resize_size))

        out_path = pathlib.Path(
            f"{CHARACTER_TRAIN_AUGMENTED_PATH}/{character_name}/{file_name}.png"
        )
        out_path.parents[0].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)


def _erosion(
    image: np.ndarray, kernel_size: int, iterations: int, erosion_size: int
) -> np.ndarray:
    elements = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )

    return cv2.erode(image, elements, iterations=iterations)


def _dilation(
    image: np.ndarray, kernel_size: int, iterations: int, dilation_size: int
) -> np.ndarray:
    elements = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size),
    )

    return cv2.dilate(image, elements, iterations=iterations)


def _deduplicate_paths(paths: List[str]) -> List[str]:
    found_hashes: Set[str] = set()
    final_paths = []

    for path in paths:
        md5 = hashlib.md5()
        with open(path, "rb") as inp:
            while True:
                data = inp.read(65536)  # 64kb
                if not data:
                    break
                md5.update(data)

        if (computed_hash := md5.hexdigest()) not in found_hashes:
            final_paths.append(path)

        found_hashes.add(computed_hash)
    logger.info(f"Removed {len(paths) - len(final_paths)} duplicates")

    return final_paths


if __name__ == "__main__":
    _load_characters(CHARACTER_TRAIN_PATH)

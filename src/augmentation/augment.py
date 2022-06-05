import hashlib
import pathlib
from glob import glob
from typing import Callable, Dict, List, Set, Tuple

import cv2
import numpy as np
import random
from cv2 import Mat

from src.utils.images import get_name, get_parent_name
from src.utils.logger import logger
from src.utils.paths import CHARACTER_TRAIN_AUGMENTED_PATH, CHARACTER_TRAIN_PATH


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


def _get_random_operation() -> Callable[[np.ndarray, int, int], np.ndarray]:
    return random.choice(
        [
            _erosion,
            _dilation,
            _shear,
        ]
    )


def augment(resize_size: int) -> None:
    characters = _load_characters(CHARACTER_TRAIN_PATH)
    n_classes: Dict[str, int] = {}
    augment_size = 300

    for character_name, _, _ in characters:
        value = n_classes.get(character_name, 0)
        n_classes[character_name] = value + 1

    for character_name, file_name, image in characters:
        image = 255 - image
        image = cv2.resize(image, (resize_size, resize_size))

        num_of_augments = augment_size // n_classes[character_name]
        for i in range(num_of_augments):
            r_image = _get_random_operation()(
                image, np.random.randint(1, 2), np.random.randint(1, 2)
            )
            out_path = pathlib.Path(
                f"{CHARACTER_TRAIN_AUGMENTED_PATH}/{character_name}/{file_name}_{i}.png"
            )
            out_path.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), r_image)


def _erosion(image: np.ndarray, iterations: int, erosion_size: int) -> np.ndarray:
    elements = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )

    return cv2.erode(image, elements, iterations=iterations)


def _dilation(image: np.ndarray, iterations: int, dilation_size: int) -> np.ndarray:
    elements = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size),
    )

    return cv2.dilate(image, elements, iterations=iterations)


def _shear(image: np.ndarray, _: int, shear_size: int) -> np.ndarray:
    return cv2.warpPerspective(
        image,
        _get_shear_matrix(shear_size),
        (image.shape[1], image.shape[0]),
    )


def _get_shear_matrix(shear_size: int) -> np.ndarray:
    return np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


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

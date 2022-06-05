import hashlib
import pathlib
from glob import glob
from typing import Callable, Dict, List, Set, Tuple
import numpy.typing as npt
import random

import cv2
import numpy as np
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


def _invert_and_resize(
    image: npt.NDArray[np.uint8], resize_size: int = 32
) -> npt.NDArray[np.uint8]:
    inverted_image = (255 - image).astype(np.uint8)
    return cv2.resize(inverted_image, (resize_size, resize_size))


def augment(augment_size: int = 500) -> None:
    characters = _load_characters(CHARACTER_TRAIN_PATH)
    n_classes: Dict[str, int] = {}

    for character_name, _, _ in characters:
        value = n_classes.get(character_name, 0)
        n_classes[character_name] = value + 1

    for character_name, character_fn, character_img in characters:
        existing_out_path = pathlib.Path(
            f"{CHARACTER_TRAIN_AUGMENTED_PATH}/{character_name}/{character_fn}.png"
        )

        existing_out_path.parents[0].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(existing_out_path), _invert_and_resize(character_img))

    for character_name, file_name, image in characters:
        image = _invert_and_resize(image)
        augments_left = augment_size - n_classes[character_name]
        augments_per_source_image = round(augments_left / n_classes[character_name])

        for i in range(augments_per_source_image):
            image_copy = image.copy()
            transformations: List[Callable[[np.ndarray], np.ndarray]] = [
                _erosion,
                _dilation,
            ]
            amount_of_tranformations = random.randint(1, len(transformations))
            chosen_operations = random.sample(transformations, amount_of_tranformations)

            for op in chosen_operations:
                image_copy = op(image_copy)

            image_copy = _shear(image_copy)  # we always shear for dupes

            out_path = pathlib.Path(
                f"{CHARACTER_TRAIN_AUGMENTED_PATH}/{character_name}/{file_name}_{i}.png"
            )
            out_path.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), image_copy)


def _erosion(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=random.randint(1, 2))


def _dilation(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=random.randint(1, 2))


def _shear(image: np.ndarray) -> np.ndarray:
    num = random.random() * 0.8 - 0.4

    return cv2.warpPerspective(
        image,
        np.array([[1, num, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        (image.shape[1], image.shape[0]),
    )


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

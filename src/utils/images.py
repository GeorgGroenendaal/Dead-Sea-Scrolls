import pathlib
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

Cmap = Literal["gray", "gist_rainbow"]


def store_image(image: np.ndarray, path: str, cmap: Cmap) -> None:
    parsed_path = pathlib.Path(path)
    parsed_path.parents[0].mkdir(parents=True, exist_ok=True)
    plt.imsave(parsed_path, image, cmap=plt.get_cmap(cmap))


def get_name(path: str) -> str:
    return pathlib.Path(path).stem

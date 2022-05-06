from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import ndimage

from src.utils.images import get_name, store_image
from src.utils.logger import logger
from src.utils.cache import memory


GrayScaleImage = npt.NDArray[np.uint8]
TraceMap = npt.NDArray[np.bool_]


class LineSegmenter:
    """
    Line segmenter based on: Handwritten Text Line Segmentation by Shredding Text into its Lines.
    (A. Nicolaou and B. Gatos) 2009
    """

    def __init__(self, binary_cutoff: int = 127, save_intermediate: bool = False):
        self.binary_cutoff = binary_cutoff
        self.save_intermediate = save_intermediate

    def segment_lines(self, image_path: str) -> None:
        name = get_name(image_path)

        image: GrayScaleImage = plt.imread(image_path)
        binarized_image = image < self.binary_cutoff

        # component labeling and height finding
        components, n_components = ndimage.label(binarized_image)
        heights = self._component_heights(components, n_components)
        mean_component_height = sum(heights) / len(heights)
        blurred_width = int(mean_component_height * 8)
        blurred_height = int(mean_component_height * 0.8)

        cached_blur = memory.cache(self._blur)
        blurred_image: GrayScaleImage = cached_blur(
            image, blurred_width, blurred_height
        )

        if self.save_intermediate:
            logger.debug(f"Storing intermediate blurred image {name}")
            store_image(blurred_image, f"data/intermediate/blurred/{name}.jpg", "gray")

    def _blur(
        self, image: GrayScaleImage, blurred_width: int, blurred_height: int
    ) -> GrayScaleImage:
        return ndimage.gaussian_filter(image, (blurred_height, blurred_width))

    def _component_heights(
        self, components: np.ndarray, n_components: int
    ) -> list[int]:
        heights: list[int] = []

        for i in range(1, n_components + 1):
            single_component = components == i
            vertical_sum = single_component.sum(axis=0)
            heights.append(vertical_sum.max())

        return heights

    def _trace(self, blurred_image: GrayScaleImage, blurred_height: int) -> TraceMap:
        map = np.zeros_like(blurred_image, dtype=np.bool_)

        return map

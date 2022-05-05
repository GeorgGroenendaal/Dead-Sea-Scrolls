import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from src.utils.images import get_name, store_image
from src.utils.logger import logger


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

        image = plt.imread(image_path)
        binarized_image = image < self.binary_cutoff

        components, n_components = ndimage.label(binarized_image)
        heights = self._component_heights(components, n_components)
        mean_heights = sum(heights) / len(heights)

        kernel_x = mean_heights * 8
        kernel_y = mean_heights * 0.8

        blurred_image = ndimage.uniform_filter(image, (kernel_y, kernel_x))

        if self.save_intermediate:
            logger.debug(f"Storing intermediate {name}")

            store_image(blurred_image, f"data/intermediate/blurred/{name}.jpg", "gray")

    def _component_heights(
        self, components: np.ndarray, n_components: int
    ) -> list[int]:
        heights: list[int] = []

        for i in range(1, n_components + 1):
            single_component = components == i
            vertical_sum = single_component.sum(axis=0)
            heights.append(vertical_sum.max())

        return heights

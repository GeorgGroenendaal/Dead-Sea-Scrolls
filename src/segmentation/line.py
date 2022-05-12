import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import ndimage

from src.utils.cache import memory
from src.utils.images import get_name, store_image
from src.utils.logger import logger

GrayScaleImage = npt.NDArray[np.uint8]
Tracers = npt.NDArray[np.int32]


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

        image: GrayScaleImage = np.array(Image.open(image_path).convert("L"))
        binarized_image = image < self.binary_cutoff

        # component labeling and height finding
        components, _ = ndimage.label(binarized_image)
        heights = self._component_heights(components)
        mean_component_height = sum(heights) / len(heights)
        blurred_width = int(mean_component_height * 8)
        blurred_height = int(mean_component_height * 0.8)

        # performing blur, reusing cache if possible
        cached_blur = memory.cache(self._blur)
        blurred_image: GrayScaleImage = cached_blur(
            image, blurred_width, blurred_height
        )

        if self.save_intermediate:
            logger.debug(f"Storing intermediate blurred image {name}")
            store_image(blurred_image, f"data/intermediate/blurred/{name}.jpg", "gray")

        tracers = self._trace(blurred_image, blurred_height)
        trace_mask_rl = self._trace_to_mask(tracers)

        # repeating for right to left
        blurred_image_flipped = self._horizontal_flip(blurred_image)
        tracers_flipped = self._trace(blurred_image_flipped, blurred_height)
        trace_mask_flipped = self._trace_to_mask(tracers_flipped)
        trace_mask_lr = self._horizontal_flip(trace_mask_flipped)

        trace_mask_combined = trace_mask_lr & trace_mask_rl

        filtered_components = self._filter_component_by_area(
            trace_mask_combined, line_height=mean_component_height
        )

        store_image(
            np.invert(filtered_components) | binarized_image,
            f"data/intermediate/trace/{name}.png",
            "gray",
        )

    def _filter_component_by_area(
        self, mask: npt.NDArray[np.bool_], line_height: float
    ) -> npt.NDArray[np.bool_]:

        cutoff_area = line_height**2  # from the paper it is 2
        cutoff_height = line_height * 2  # we added this
        components, _ = ndimage.label(mask)

        for loc in ndimage.find_objects(components):
            single_component = components[loc]
            area = np.count_nonzero(single_component)
            height = np.max(np.count_nonzero(single_component, 0))

            if area < cutoff_area or height < cutoff_height:
                components[loc] = 0

        return components > 0

    def _blur(
        self, image: GrayScaleImage, blurred_width: int, blurred_height: int
    ) -> GrayScaleImage:
        return ndimage.gaussian_filter(image, (blurred_height, blurred_width))

    def _component_heights(self, labeled_components: np.ndarray) -> list[int]:
        heights: list[int] = []

        for loc in ndimage.find_objects(labeled_components):
            single_component = labeled_components[loc] != 0
            vertical_sum = single_component.sum(axis=0)
            heights.append(vertical_sum.max())

        return heights

    def _trace(self, blurred_image: GrayScaleImage, blurred_height: int) -> Tracers:
        height, width = blurred_image.shape
        tracers = np.zeros_like(blurred_image, dtype=np.int32)
        tracers[:, 0] = np.arange(height)

        offset = blurred_height // 2

        for i in range(1, width):
            previous = tracers[:, i - 1]
            indices_up = np.clip(previous - offset, 0, height - 1)
            indices_down = np.clip(previous + offset, 0, height - 1)

            values_up = blurred_image[indices_up, i]
            values_down = blurred_image[indices_down, i]

            tracers[:, i] = np.where(values_up > values_down, previous - 1, previous)
            tracers[:, i] = np.where(
                values_up < values_down, previous + 1, tracers[:, i]
            )

        return tracers

    def _trace_to_mask(self, tracers: Tracers) -> npt.NDArray[np.bool_]:
        height, width = tracers.shape
        mask = np.zeros_like(tracers, dtype=np.bool_)

        for i in range(width):
            indices = np.arange(height)
            tracer_col = tracers[:, i]
            mask[:, i] = np.invert(np.isin(indices, tracer_col))

        return mask

    def _horizontal_flip(self, arr: np.ndarray) -> np.ndarray:
        return arr[:, ::-1]

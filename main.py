from email.policy import default
import glob
from typing import Optional, Union

import click
from tqdm.contrib.concurrent import process_map

from src.augmentation.augment import augment
from src.segmentation.character import CharacterSegmenter
from src.segmentation.line import LineSegmenter

# from src.augmentation.augmentation import ImageAugmentation
from src.utils.logger import logger
from src.utils.paths import LINE_SEGMENT_PATH
from src.utils.zip import unzip_all
from src.classification.classifier import Classifier


@click.group()
def cli() -> None:
    pass


@cli.command()
def prepare() -> None:
    logger.info("Unzipping data")
    unzip_all("data/raw/image-data.zip", "data/unpacked")
    unzip_all("data/raw/characters.zip", "data/unpacked")
    logger.info("Done")


@cli.command()
@click.option("--debug/--no-debug", default=False)
@click.option("--file", default=None)
def linesegment(debug: bool, file: Union[str, None]) -> None:
    line_segmenter = LineSegmenter(debug=debug)

    if file:
        logger.info(f"Starting line segmentation on {file}")
        line_segmenter.segment_lines(file)
    else:
        logger.info("Starting line segmentation on all binarized images")
        binary_files = glob.glob("data/unpacked/image-data/*binarized.jpg")

        # concurrent processing
        process_map(line_segmenter.segment_lines, binary_files)


@cli.command()
@click.option("--file", default=None)
@click.option("--debug/--no-debug", default=False)
def charactersegment(file: Union[str, None], debug: bool = False) -> None:
    logger.info("Starting character segmentation")
    character_segmenter = CharacterSegmenter(debug=debug)

    if file:
        character_segmenter.segment_characters(file)
    else:
        files = glob.glob(f"{LINE_SEGMENT_PATH}/**/*.png")

        if not files:
            logger.warning("No images with segmented lines, did you run linesegment?")

        for file in files:
            character_segmenter.segment_characters(file)


@cli.command(name="augment")
@click.option("--resize_size", default=32)
def run_augment(resize_size: int) -> None:
    augment(resize_size)


if __name__ == "__main__":
    cli()

import glob
from typing import Union

import click
from tqdm.contrib.concurrent import process_map

from src.segmentation.line import LineSegmenter
from src.augmentation.augmentation import ImageAugmentation
from src.utils.logger import logger
from src.utils.zip import unzip_all
from src.classification.classifier import Classifier
import os


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
@click.option("--save-intermediate/--no-save-intermediate", default=False)
@click.option("--file", default=None)
def segment(save_intermediate: bool, file: Union[str, None]) -> None:
    line_segmenter = LineSegmenter(save_intermediate=save_intermediate)

    if file:
        logger.info(f"Starting line segmentation on {file}")
        line_segmenter.segment_lines(file)
    else:
        logger.info("Starting line segmentation on all binarized images")
        binary_files = glob.glob("data/unpacked/image-data/*binarized.jpg")

        # concurrent processing
        process_map(line_segmenter.segment_lines, binary_files)


@cli.command()
@click.option("--folder", default=None)
@click.option("--train", default=None)
@click.option("--predict", default=None)
def augment(folder: str) -> None:
    if not folder:
        folder = "data/unpacked/image-data"

    # check if data path exist stop if not
    if not os.path.exists(folder):
        logger.error("Data path does not exist, please run prepare command first")
        return

    logger.info("Starting image augmentation")

    ImageAugmentation(folder)


@cli.command()
@click.option("--folder", default=None)
@click.option("--train", default=None)
@click.option("--predict", default=None)
def classify(folder: str, train: bool = False, predict: str = None) -> None:

    if train:
        logger.info("Starting training")
        folder = "data/unpacked/characters"

        # check if data path exist stop if not
        if not os.path.exists(folder):
            logger.error("Data path does not exist, please run prepare command first")
            return

        logger.info("Starting classification")

        Classifier(train=True)

    if predict:
        logger.info("Starting prediction {}".format(predict))
        if not os.path.exists(predict):
            logger.error("Image does not exist")
            return
        Classifier(predicit=True, predict_image=predict)


if __name__ == "__main__":
    cli()

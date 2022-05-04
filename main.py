import glob

import click
from tqdm import tqdm

from src.segmentation.line import LineSegmenter
from src.utils.logger import logger
from src.utils.zip import unzip_all


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
def segment(save_intermediate: bool) -> None:
    binary_files = glob.glob("data/unpacked/image-data/*binarized.jpg")
    line_segmenter = LineSegmenter(save_intermediate=save_intermediate)

    logger.info("Starting line segmentation")

    for file in tqdm(binary_files):
        line_segmenter.segment_lines(file)


if __name__ == "__main__":
    cli()

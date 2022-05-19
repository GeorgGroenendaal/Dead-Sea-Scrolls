import glob

import click
from tqdm.contrib.concurrent import process_map

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
@click.option("--file", default=None)
def segment(save_intermediate: bool, file: str | None) -> None:
    line_segmenter = LineSegmenter(save_intermediate=save_intermediate)

    if file:
        logger.info(f"Starting line segmentation on {file}")
        line_segmenter.segment_lines(file)
    else:
        logger.info("Starting line segmentation on all binarized images")
        binary_files = glob.glob("data/unpacked/image-data/*binarized.jpg")

        # concurrent processing
        process_map(line_segmenter.segment_lines, binary_files)


if __name__ == "__main__":
    cli()

import glob
import pathlib
from typing import Union

import click
import numpy as np
from tqdm.contrib.concurrent import process_map

from src.augmentation.augment import augment
from src.classification.classifier import Classifier
from src.segmentation.character import CharacterSegmenter
from src.segmentation.line import LineSegmenter
from src.utils.font import text_to_font
from src.utils.images import get_name
from src.utils.logger import logger
from src.utils.paths import LINE_SEGMENT_PATH
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
@click.option("--debug/--no-debug", default=True)
@click.option("--file", default=None)
def linesegment(debug: bool, file: Union[str, None]) -> None:
    line_segmenter = LineSegmenter(debug=debug)

    if file:
        logger.info(f"Starting line segmentation on {file}")
        line_segmenter.segment_from_path(file)
    else:
        logger.info("Starting line segmentation on all binarized images")
        binary_files = glob.glob("data/unpacked/image-data/*binarized.jpg")

        # concurrent processing
        process_map(line_segmenter.segment_from_path, binary_files)


@cli.command()
@click.option("--file", default=None)
@click.option("--debug/--no-debug", default=True)
def charactersegment(file: Union[str, None], debug: bool = False) -> None:
    logger.info("Starting character segmentation")
    character_segmenter = CharacterSegmenter(debug=debug)

    if file:
        character_segmenter.segment_from_path(file)
    else:
        files = glob.glob(f"{LINE_SEGMENT_PATH}/**/*.png")

        if not files:
            logger.warning("No images with segmented lines, did you run linesegment?")

        for file in files:
            character_segmenter.segment_from_path(file)


@cli.command(name="augment")
def run_augment() -> None:
    augment()


@cli.command()
@click.option("--train/--no-train", default=True)
@click.argument("name")
def train(train: bool, name: str) -> None:
    Classifier(train=train, model_filename=name, debug=True)


@cli.command()
@click.argument("directory", default="data/unpacked/image-data/")
@click.option("--o", "out_dir", default="results/")
@click.option("--suffix", default="")
def run(directory: str, out_dir: str, suffix: str) -> None:
    classifier = Classifier(train=False, model_filename="augmented_cnn", debug=True)
    line_segmenter = LineSegmenter()
    character_segmenter = CharacterSegmenter(min_distance=40)

    binary_files = glob.glob(f"{directory}/*{suffix}.jpg")
    output = ""
    for file in binary_files:
        out_name = get_name(file)
        logger.info(f"Processing {file}")
        lines = line_segmenter.segment_from_path(file)

        for line in lines:
            characters = character_segmenter.segment(line)
            if characters:
                stacked_characters = np.stack(characters, axis=0)
                proba = classifier.predict_batch(stacked_characters.astype(np.int64))
                line_predictions = classifier.decode_proba_batch(proba)

                for character, _ in line_predictions:
                    output += f"{character} "

                output += "\n"

        out_path = pathlib.Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(pathlib.Path(out_path, out_name + ".txt"), "w") as out:
            mapped_output = text_to_font(output)
            out.write(mapped_output)
            output = ""


if __name__ == "__main__":
    cli()

# Dead-Sea-Scrolls


## Prerequisites

- Python ^3.8 and < 3.11
- [Poetry](https://python-poetry.org/) for dependencies

For development
- (Optionally) Docker 
- (Optionally) Vscode with devcontainer extension


### Development

Development is easiest inside a devcontainer, and we strongly recommend to use it. Install the extension and open the project inside a container. Then execute the following commands.

    poetry config virtualenvs.in-project true
    poetry install


Activate the environment and execute the program by.


    poetry shell
    python main.py

The following comands can be used for different steps in pipline.
Commands:
  augment
  charactersegment
  linesegment
  prepare
  run
  train

1. Prepare          - prepares the data by unzipping the data in the correct folder - python main.py prepare
2. Augment          - Augments the data with different augmentation methods (erosion, dilation, shearing), takes optional parameter (--resize_size, default=32) 
3. Train            - Train the classifier, takes optional parameter (--train or --no-train, default=True) for training on augmented data
4. Run              - Run the model on the image-data, takes optional parameters:
                            <dir-to-data> <default="data/unpacked/image-data/">, 
                            <--output_dir> <default="results/">, 
                            <--suffix> <default="">
5. Linesegment      - Segments the linses on the given file, python main.py linesegment --file=<path_to_file>
6. Charactersegment - Segments the lines into characters with python main.py charachtersegment --file=<path_to_file>
   




# Dead-Sea-Scrolls

## Prerequisites

- Python ^3.8 and < 3.11
- (Optional) [Poetry](https://python-poetry.org/) for dependencies

For development
- (Optionally) Docker 
- (Optionally) Vscode with devcontainer extension

### Setup

    pip install -r requirements.txt
    # or using poetry
    poetry install
    poetry shell


### Performing full pipeline

*NOTE*: Print help using `python main.py --help`

**IMPORTANT**. This program only works well on binarized images.

The main command will read all `.jpg` images from `INPUT-DIRCTORY` and output the .txt files in the `results` folder. To do this, execute the following command from the root of the project.

    python main.py run [INPUT-DIRECTORY]

For example:

    python main.py run test-images

The output directory can be changed using the `--o` flag. Defaults to `results`.

    python main.py run test-images --o out/folder
    
To only select specific images it is possible to use the `--suffix` flag. This will only select `.jpg` files that end with a specific string. Usefull for selecting only binarized images.

    python main.py run test-images --suffix binarized


### Other commands

Not usefull for evaluation, but other commands can be used to train, segment and perform augmentation on the dataset.

See the full list of commands using:

    python main.py --help

Get instructions for a specific command using

    python main.py [command] --help
    # for example
    python main.py run --help

- **Prepare**: prepares the data by unzipping the data in the correct folder
- **Augment**: Augments the data with different augmentation methods, (dilate, )
- **Train**: Trains the classifier on the AUGMENTED dataset
- **Run**: the model on the image-data
- **Linesegment** Segments into lines
- **Charactersegment** - Segments the lines into characters

### Development

Development is easiest inside a devcontainer, and we strongly recommend to use it. Install the extension and open the project inside a container. Then execute the following commands.

    poetry config virtualenvs.in-project true
    poetry install


Activate the environment and execute the program by.


    poetry shell
    python main.py


# Dead-Sea-Scrolls


## Prerequisites

- Python >= 3.10
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

import zipfile
from pathlib import Path


def unzip_all(file: str, dest_directory: str) -> None:
    Path(dest_directory).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(file, "r") as f:
        f.extractall(dest_directory)

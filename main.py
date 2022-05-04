import click

from src.utils.zip import unzip_all


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug: bool) -> None:
    pass


@cli.command()
def prepare() -> None:
    click.echo("Unzipping data")
    unzip_all("data/raw/image-data.zip", "data/unpacked")
    unzip_all("data/raw/characters.zip", "data/unpacked/characters")


if __name__ == "__main__":
    cli()

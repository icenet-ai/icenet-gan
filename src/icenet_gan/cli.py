"""Console script for icenet_gan."""
import icenet_gan

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for icenet_gan."""
    console.print("Replace this message by putting your code into "
               "icenet_gan.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()

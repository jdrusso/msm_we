"""Console script for msm_we."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for msm_we."""
    click.echo("Replace this message by putting your code into "
               "msm_we.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

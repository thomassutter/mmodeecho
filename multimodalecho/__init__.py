
from ctypes import util
import click

from multimodalecho.__version__ import __version__
import multimodalecho.datasets as datasets
import multimodalecho.utils as utils
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@click.group()
def main():
    """Entry point for command line interface."""


del click

main.add_command(utils.run_model.run)
main.add_command(utils.extract_Mmode.run)
main.add_command(utils.extract_videos.run)
main.add_command(utils.collect_results.run)
main.add_command(utils.epoch_plots.run)

__all__ = ["__version__", "datasets", "main", "utils"]

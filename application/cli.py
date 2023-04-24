from pathlib import Path

import typer as typer

from config import MASK_THRESHOLD
from model.model import MyDetectionSegmentationModel
from service.model_service import ModelService

model_service = ModelService()

cli_app = typer.Typer()


@cli_app.command()
def demo(file_path):
    """
    Demonstrates the trained model on a given file.

    :param file_path: The path of the file to be processed.

    :return: None
    """
    model_service.demo_cli(file_path)


@cli_app.command()
def train(dataset_path: str):
    """
    Trains the model on the specified dataset.

    :param dataset_path: The path of the dataset directory.

    :return: None
    """
    model = MyDetectionSegmentationModel(MASK_THRESHOLD)
    model.train(Path(dataset_path))


@cli_app.command()
def evaluate(dataset_path: str):
    """
    Evaluates the performance of the trained model on the specified dataset.

    :param dataset_path: The path of the dataset directory.

    :return: None
    """
    model = MyDetectionSegmentationModel(MASK_THRESHOLD)
    model.warmup()
    model.evaluate(dataset_path)


if __name__ == '__main__':
    cli_app()

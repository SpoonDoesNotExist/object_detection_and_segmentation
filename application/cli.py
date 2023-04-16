import logging

import typer as typer

from service.model_service import ModelService

model_service = ModelService()

cli_app = typer.Typer()


@cli_app.command()
def demo(file_path):
    model_service.demo_cli(file_path)


@cli_app.command()
def train(dataset):
    raise NotImplementedError()


@cli_app.command()
def evaluate(dataset):
    raise NotImplementedError()


if __name__ == '__main__':
    cli_app()

import typer as typer

from application.model.model import MyDetectionSegmentationModel

det_seg_model = MyDetectionSegmentationModel()
det_seg_model.warmup()

cli_app = typer.Typer()


@cli_app.command()
def demo(file):
    raise NotImplementedError()


@cli_app.command()
def train(dataset):
    raise NotImplementedError()


@cli_app.command()
def evaluate(dataset):
    raise NotImplementedError()


if __name__ == '__main__':
    cli_app()

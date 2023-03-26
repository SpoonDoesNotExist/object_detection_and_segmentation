import logging

from .base_model import BaseModel


class MyDetectionSegmentationModel(BaseModel):

    def __init__(self, logger=logging.getLogger()):
        super().__init__()
        self.logger = logger

    def process(self, frame):
        return frame

    def warmup(self):
        pass

    def evaluate(self, dataset):
        pass

    def demo(self, file_path):
        pass

    def train(self, dataset):
        pass

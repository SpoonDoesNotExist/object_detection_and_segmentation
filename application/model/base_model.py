from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Defines interface for all models"""

    def __init__(self):
        pass

    @abstractmethod
    def train(self, dataset):
        """Performs model training.

        :param dataset:
        :return:
        """

    @abstractmethod
    def evaluate(self, dataset):
        """Evaluates the model with the provided dataset

        :param dataset:
        :return:
        """

    @abstractmethod
    def demo(self):
        """Runs real-time demo with provided image or video file."""

    @abstractmethod
    def warmup(self):
        """Reloads trained model"""

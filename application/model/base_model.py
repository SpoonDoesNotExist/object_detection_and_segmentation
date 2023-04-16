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

        :param dataset: Path to dataset
        :return:
        """

    @abstractmethod
    def process(self, frame):
        """Process one frame

        :param frame: numpy array
        :return:
        """

    @abstractmethod
    def warmup(self):
        """Reloads trained model"""

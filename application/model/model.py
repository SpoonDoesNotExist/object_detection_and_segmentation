import logging

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from config import MODEL_PATH
from .arcitecture import UNet
from .base_model import BaseModel
from .dataset import CloddDataset


class MyDetectionSegmentationModel(BaseModel):
    """
    A custom detection and segmentation model for identifying contours in images using a UNet architecture.
    """

    def __init__(self, mask_threshold, logger=None):
        """

        :param mask_threshold:  A threshold value for binarizing the output mask.
        :param logger: A logger object for logging messages.
        """
        super().__init__()
        self.mask_threshold = mask_threshold
        self.model = UNet(n_channels=3, n_classes=1, out_channels=8, upsample=False)

        if logger is None:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger()
        else:
            self.logger = logger

    def process(self, frame):
        """
        Process a single frame of an image.

        :param frame: A single frame of an image.

        :return: A tuple of the processed frame and a list of contour arrays.
        """
        frame = self.__preprocess_frame(frame)
        with torch.no_grad():
            input_data = torch.from_numpy(frame.transpose(-1, 0, 1)).unsqueeze(0).type(torch.float32)
            probs = self.model(input_data).sigmoid()

        mask = probs.numpy().squeeze() > self.mask_threshold
        contours = self.__get_contours(mask)
        return frame, contours

    def warmup(self):
        """
        Load the model from a saved file and prepare it for inference

        :return: None
        """
        self.model = torch.jit.load(MODEL_PATH)
        self.model.eval()

    def evaluate(self, dataset_path):
        """
        Evaluate the model on a dataset.

        :param dataset_path: The dataset path to evaluate the model on.
        :return: None
        """
        cds = CloddDataset(dataset_path / 'result.json', ['big_clod'])

        valid_dataloader = DataLoader(cds, batch_size=16, shuffle=False)

        trainer = pl.Trainer(gpus=1, max_epochs=30, log_every_n_steps=10)
        trainer.test(self.model, valid_dataloader)

    def train(self, dataset_path):
        """
        Train the model on a dataset.

        :param dataset_path: The dataset path to train the model on.

        :return: None
        """
        cds = CloddDataset(dataset_path / 'result.json', ['big_clod'])

        train_dataset = torch.utils.data.Subset(cds, range(140))
        valid_dataset = torch.utils.data.Subset(cds, range(140, 140 + 42))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

        trainer = pl.Trainer(gpus=1, max_epochs=30, log_every_n_steps=10, accelerator='cpu')
        trainer.fit(self.model, train_dataloader, valid_dataloader)

    def __get_contours(self, segmentation_mask):
        """
        Extract the contours from a binary segmentation mask.

        :param segmentation_mask: A binary segmentation mask.

        :return: A list of contour arrays.
        """
        segmentation_mask = np.uint8(segmentation_mask)
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def __preprocess_frame(self, frame):
        """
        Preprocess a single frame.

        :param frame: A single frame to preprocess.

        :return: Preprocessed frame.
        """
        frame = cv2.convertScaleAbs(frame, alpha=3, beta=0)
        frame = cv2.resize(
            frame,
            (frame.shape[1] // 2, frame.shape[0] // 2)
        )
        return frame

import logging

import cv2
import numpy as np
import torch

from config import MODEL_PATH

from . import arcitecture
from .base_model import BaseModel


class MyDetectionSegmentationModel(BaseModel):

    def __init__(self, logger=logging.getLogger()):
        super().__init__()
        self.model = arcitecture.PetModel("unetplusplus", "efficientnet-b0", in_channels=3, out_classes=1)
        self.logger = logger

    def process(self, frame):
        if self.model is None:
            raise Exception('Load model!')

        frame = self.__preprocess_frame(frame)
        with torch.no_grad():
            probs = self.model(frame).sigmoid()

        frame, mask = frame.numpy().squeeze().transpose(1, 2, 0), probs.numpy().squeeze() > 0.5
        contours = self.__get_contours(mask)
        return frame, contours

    def warmup(self):
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

    def evaluate(self, dataset):
        pass

    def train(self, dataset):
        pass

    def __get_contours(self, segmentation_mask):
        segmentation_mask = np.uint8(segmentation_mask)
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def __preprocess_frame(self, frame):
        frame = self.__crop_to_correct_shape(frame)
        # brightness

        return frame

    def __crop_to_correct_shape(self, frame):
        height, width, channels = frame.shape
        h_diff, w_diff = height % 32, width % 32
        cropped_frame = frame

        if h_diff != 0:
            left_diff, right_diff = self.__get_crop_size(h_diff)
            cropped_frame = cropped_frame[left_diff:-right_diff, :, :]
        if w_diff != 0:
            left_diff, right_diff = self.__get_crop_size(w_diff)
            cropped_frame = cropped_frame[:, left_diff:-right_diff, :]

        cropped_frame = cropped_frame.transpose(-1, 0, 1)
        return torch.from_numpy(cropped_frame).unsqueeze(0)

    def __get_crop_size(self, diff):
        return int(np.ceil(diff / 2.)), int(np.floor(diff / 2.))

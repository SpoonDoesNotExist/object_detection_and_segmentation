import logging

import cv2
import numpy as np

from config import TRAIN_METRICS_PATH, TEST_METRICS_PATH, DRAW_COLOR, DRAW_COLOR_MAX, CONTOUR_AREA_THRESHOLD, \
    MASK_THRESHOLD
from model.model import MyDetectionSegmentationModel


class ModelService:
    def __init__(self, logger=None):
        """
        :param logger: A logging object to use for logging, defaults to None
        """
        self.logger = logger
        if self.logger is None:
            self.logger = self.__create_logger('default_logger', logging.DEBUG)

        self.model = MyDetectionSegmentationModel(MASK_THRESHOLD, self.logger)
        self.model.warmup()

        self.color = DRAW_COLOR
        self.contour_area_threshold = CONTOUR_AREA_THRESHOLD

    def demo(self, file_path):
        """Open video file and stream it by yield

        :param file_path: The path to the video file to open
        :return: A generator that yields JPEG-encoded video frames as bytes
        """

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(file_path)

        # Loop through the video frames
        while cap.isOpened():
            print('proc frame...')
            # Read the next frame from the video file
            success, frame = cap.read()

            frame = cv2.convertScaleAbs(frame, alpha=3, beta=0)
            frame = cv2.resize(
                frame,
                (frame.shape[1] // 2, frame.shape[0] // 2)
            )

            # If there was an error reading the frame, log it and break out of the loop
            if not success:
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.logger.info(f'Failed to read frame number {frame_number}')
                break

            # Process the frame using model
            frame, clod_contours = self.model.process(frame)
            if len(clod_contours) > 0:
                self.__draw_contours(frame, clod_contours)
                self.__draw_max_contour(frame, clod_contours)
            else:
                self.logger.info(f'No clodds found')

            yield self.__get_jpg_bytes(frame)

    def demo_cli(self, file_path):
        """
        Open video file and stream it as application.

        :param file_path: The path to the video file to open

        :return: None
        """
        cap = cv2.VideoCapture(file_path)

        frame_area = None
        while cap.isOpened():
            success, frame = cap.read()

            # Press Q on keyboard to  exit
            if (cv2.waitKey(25) & 0xFF == ord('q')) or not success:
                break

            # Process the frame using model
            frame, clod_contours = self.model.process(frame)

            if frame_area is None:
                frame_area = cv2.countNonZero(np.ones(frame.shape[:2]))

            if len(clod_contours) > 0:
                areas = self.__calc_areas(clod_contours, frame_area)
                clod_contours = self.__filter_contours(clod_contours, areas)

                self.__draw_contours(frame, clod_contours, DRAW_COLOR)
                self.__draw_max_contour(frame, clod_contours)
            else:
                print(f'No clodds found')

            cv2.imshow('Frame', frame)

    def __draw_contours(self, frame, contours, color, thickness=2):
        """
        Draw contours on the given frame with the specified color.

        :param frame: The frame on which to draw the contours
        :param contours: The list of contours to draw
        :param color: The color with which to draw the contours

        :return: None
        """
        cv2.drawContours(frame, contours, -1, color, thickness)

    def __calc_areas(self, contours, full_area):
        """
        Calculate the areas of the given contours.

        :param contours: The list of contours whose areas to calculate
        :param full_area: The full area of the frame

        :return: A numpy array containing the areas of the contours
        """
        return np.fromiter(map(
            cv2.contourArea,
            contours
        ), dtype=np.float64) / full_area

    def __draw_max_contour(self, frame, contours):
        """
        Draw the maximum contour on the given frame.

        :param frame: The frame on which to draw the maximum contour
        :param contours:  The list of contours from which to find the maximum contour

        :return: None
        """
        frame_area = cv2.countNonZero(np.ones(frame.shape[:2]))

        areas = self.__calc_areas(contours, frame_area)
        contours = self.__filter_contours(contours, areas)

        max_idx = np.argmax(areas)
        max_area = areas[max_idx]

        self.__add_text(frame, f'Max area: {max_area:.5f}')
        self.__add_text(frame, f"Clods count: {len(contours)}", 40, 20)
        self.__draw_contours(frame, [contours[max_idx]], DRAW_COLOR_MAX, -1)

    def __get_jpg_bytes(self, frame):
        """
        Convert the given frame to a JPEG-encoded byte string.

        :param frame: The frame to convert to a JPEG-encoded byte string

        :return: The JPEG-encoded byte string
        """
        processed_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        return b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n'

    def __save_metrics(self, train_metrics, test_metrics):
        """
        Save the training and test metrics to file.

        :param train_metrics: The training metrics to save
        :param test_metrics: The test metrics to save

        :return: None
        """
        with open(TRAIN_METRICS_PATH, 'w') as f:
            f.write(str(train_metrics))
        with open(TEST_METRICS_PATH, 'w') as f:
            f.write(str(test_metrics))

    def __add_text(self, img, text, top_margin=10, right_margin=20):
        """
        Add text to the top right corner of an image.

        :param img: The image to which to add text
        :param text: The text to add to the image
        :param top_margin: The margin from the top of the frame
        :param right_margin: The margin from the right of the frame

        :return: The image with the text added
        """
        # Set font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Get text size
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Set text position
        text_x = img.shape[1] - text_size[0] - right_margin  # 10 pixels from right edge
        text_y = text_size[1] + top_margin  # 10 pixels from top edge

        # Add text to image
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        return img

    def __filter_contours(self, contours, areas):
        """
        Filter the given contours by area.

        :param contours: The list of contours to filter
        :param areas: The areas of the contours

        :return: The filtered list of contours
        """
        indices = [i for i, x in enumerate(areas) if x >= self.contour_area_threshold]
        if len(indices) == 0:
            return contours
        return [contours[i] for i in indices]

    def __create_logger(self, name, level=logging.INFO):
        """
        Create a logger object.

        :param name: The name of the logger
        :param level: The logging level, defaults to logging.INFO

        :return: The logger object
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create a console handler and set its log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter and add it to the console handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)

        return logger

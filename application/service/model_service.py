import cv2

from config import TRAIN_METRICS_PATH, TEST_METRICS_PATH
from model.model import MyDetectionSegmentationModel


class ModelService:
    def __init__(self, logger):
        self.logger = logger
        self.model = MyDetectionSegmentationModel(self.logger)

    def demo(self, file_path):
        """Open video file and stream it by yield

        :param file_path: The path to the video file to open
        :return: A generator that yields JPEG-encoded video frames as bytes
        """

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(file_path)

        # Loop through the video frames
        while cap.isOpened():
            # Read the next frame from the video file
            success, frame = cap.read()

            # If there was an error reading the frame, log it and break out of the loop
            if not success:
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.logger.info(f'Failed to read frame number {frame_number}')
                break

            # Process the frame using model
            frame = self.model.process(frame)

            # Convert the frame to a JPEG-encoded byte string
            frame = self.__get_jpg_bytes(frame)

            # Yield the JPEG-encoded frame as a MIME multipart message with type "image/jpeg"
            yield b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

    def __get_jpg_bytes(self, frame):
        return cv2.imencode('.jpg', frame)[1].tobytes()

    def __save_metrics(self, train_metrics, test_metrics):
        with open(TRAIN_METRICS_PATH, 'w') as f:
            f.write(str(train_metrics))
        with open(TEST_METRICS_PATH, 'w') as f:
            f.write(str(test_metrics))

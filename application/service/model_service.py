import cv2

from config import TRAIN_METRICS_PATH, TEST_METRICS_PATH


class ModelService:
    def __init__(self, logger):
        self.logger = logger

    def demo(self, file_path):
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to a JPEG image
            frame = cv2.imencode('.jpg', frame)[1].tobytes()

            # Yield the frame as a Flask response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def _save_metrics(self, train_metrics, test_metrics):
        with open(TRAIN_METRICS_PATH, 'w') as f:
            f.write(str(train_metrics))
        with open(TEST_METRICS_PATH, 'w') as f:
            f.write(str(test_metrics))

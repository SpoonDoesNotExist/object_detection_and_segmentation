from application.config import TRAIN_METRICS_PATH, TEST_METRICS_PATH


class ModelService:
    def __init__(self, app):
        self.app = app

    def _save_metrics(self, train_metrics, test_metrics):
        with open(TRAIN_METRICS_PATH, 'w') as f:
            f.write(str(train_metrics))
        with open(TEST_METRICS_PATH, 'w') as f:
            f.write(str(test_metrics))

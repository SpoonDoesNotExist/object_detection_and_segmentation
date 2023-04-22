from config import DOCKER_BUILD_DATETIME, TRAIN_METRICS_PATH, TEST_METRICS_PATH


class InfoService:
    def __init__(self, logger):
        """
        :param logger:  An instance of a logger object to log messages.
        """
        self.logger = logger

    def get_credentials(self):
        """
        Returns a dictionary with the author's credentials, Docker build datetime, and model metrics.
        """
        docker_datetime = self._read_data(DOCKER_BUILD_DATETIME)
        train_metrics = self._read_data(TRAIN_METRICS_PATH)
        test_metrics = self._read_data(TEST_METRICS_PATH)

        return {
            'Name': 'Eduard',
            'Surname': 'Khusnutdinov',
            'University': {
                'Name': 'Tomsk State University',
                'Grade': '3',
            },
            'Docker build datetime': docker_datetime,
            'Train metrics': train_metrics,
            'Test metrics': test_metrics,
        }

    def _read_data(self, path):
        """
        Reads data from the specified file.

        :param path: The path to the file to read.

        :return: The contents of the file as a string.
        """
        with open(path, 'r') as f:
            return ' '.join(
                f.readlines()
            )

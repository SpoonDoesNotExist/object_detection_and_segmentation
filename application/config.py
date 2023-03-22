from pathlib import Path

HOST = '0.0.0.0'
DEFAULT_PORT = 5000
DEV_RUN_MODE = 'dev'
PROD_RUN_MODE = 'production'
APPLICATION_ROOT = '/api'

DOCKER_BUILD_DATETIME = 'application/build_date.txt'

ROOT_PATH = Path('application')
DATA_PATH = ROOT_PATH / 'data'

LOGING_PATH = DATA_PATH / 'log_file.log'

TRAIN_METRICS_PATH = DATA_PATH / 'train_metrics.txt'
TEST_METRICS_PATH = DATA_PATH / 'test_metrics.txt'

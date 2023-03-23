from pathlib import Path

HOST = '0.0.0.0'
DEFAULT_PORT = 5000
DEV_RUN_MODE = 'dev'
PROD_RUN_MODE = 'production'
APPLICATION_ROOT = '/api'

DOCKER_BUILD_DATETIME = 'build_date.txt'

ROOT_PATH = Path('.')
DATA_PATH = ROOT_PATH / 'data'

CONTROLLER_BASE_DATA = Path('../data')
UPLOAD_FOLDER = CONTROLLER_BASE_DATA / 'upload_files'
TEMPLATE_FOLDER = CONTROLLER_BASE_DATA / 'templates'

LOGING_PATH = DATA_PATH / 'logs' / 'log_file.log'

TRAIN_METRICS_PATH = CONTROLLER_BASE_DATA / 'metrics' / 'train_metrics.txt'
TEST_METRICS_PATH = CONTROLLER_BASE_DATA / 'metrics' / 'test_metrics.txt'

VIDEO_TEMPLATE_NAME = 'video.html'

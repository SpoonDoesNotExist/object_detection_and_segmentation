import os
from logging.handlers import RotatingFileHandler

from flask import Flask
import logging

from config import DEFAULT_PORT, DEV_RUN_MODE, PROD_RUN_MODE, APPLICATION_ROOT, LOGING_PATH, HOST, \
    UPLOAD_FOLDER, TEMPLATE_FOLDER


def check_env_vars():
    """Check all environment variables"""

    global app
    global run_mode
    global port

    # Get run mode
    run_mode = os.environ.get('run_mode', DEV_RUN_MODE)
    message = f'Run mode {run_mode}'
    app.logger.info(message)

    # Get port
    port = os.environ.get('port', DEFAULT_PORT)
    if port == DEFAULT_PORT:
        if run_mode == DEV_RUN_MODE:
            app.logger.info(f'Application port is not set. Using default {port}')
        elif run_mode == PROD_RUN_MODE:  # Do not run in production with default port
            app.logger.info(f'Set up application port')
            raise KeyError(message)
    else:
        app.logger.info(f'Using port {port}')


def init_logging():
    """Initialise app logging"""

    global app
    global logging_path

    # Create logging file. Clean up if exists
    logfile = open(logging_path, 'w')
    logfile.close()

    # Define formatting pattern
    formatter = logging.Formatter(
        fmt='>>> %(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler = RotatingFileHandler(logging_path)
    handler.setFormatter(formatter)

    default_logger = logging.getLogger()
    default_logger.setLevel(logging.INFO)

    app.logger.addHandler(handler)


run_mode = DEV_RUN_MODE
host = HOST
port = DEFAULT_PORT
logging_path = LOGING_PATH

app = Flask(
    __name__,
    static_folder=UPLOAD_FOLDER,
    template_folder=TEMPLATE_FOLDER
)

app.config['APPLICATION_ROOT'] = APPLICATION_ROOT

from .info import info_page
from .model import model_page

app.register_blueprint(model_page, url_prefix='/api')
app.register_blueprint(info_page, url_prefix='/api')

init_logging()
check_env_vars()

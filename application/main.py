from config import DEV_RUN_MODE, PROD_RUN_MODE
from controller.app import app, port, host, run_mode
from waitress import serve

if __name__ == '__main__':
    if run_mode == DEV_RUN_MODE:
        app.run(host=host, port=port)
    elif run_mode == PROD_RUN_MODE:
        serve(app, host=host, port=port)
    else:
        raise NotImplementedError(f'Unexpected run mode: {run_mode}')

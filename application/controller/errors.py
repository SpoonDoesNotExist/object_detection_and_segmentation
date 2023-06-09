from .app import app


@app.errorhandler(500)
def internal_server_error(e):
    app.logger.info(str(e))
    return f'{str(e)}', 500

from flask import jsonify

from .app import app
from application.service.info_service import InfoService

info_service = InfoService(app)


@app.route('/info', methods=['GET'])
def info():
    return jsonify(info_service.get_credentials())

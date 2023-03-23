from flask import jsonify, Blueprint

from controller.app import app
from service.info_service import InfoService

info_page = Blueprint('info_page', __name__)

info_service = InfoService(app.logger)


@info_page.route('/info', methods=['GET'])
def info():
    return jsonify(info_service.get_credentials())

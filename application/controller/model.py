import os

from flask import request, flash, redirect, Blueprint, render_template, Response, make_response, abort
from werkzeug.utils import secure_filename

from config import VIDEO_TEMPLATE_NAME, UPLOAD_TEMPLATE_NAME, VIDEO_FEED_MIMETYPE, ERROR_TEMPLATE_NAME
from controller.app import app
from service.model_service import ModelService

model_page = Blueprint('model_page', __name__)
model_service = ModelService(app.logger)


@model_page.route('/index')
def index():
    return render_template(UPLOAD_TEMPLATE_NAME)


@model_page.route('/demo/<filename>')
def demo(filename):
    file_path = f'{app.static_folder}/{filename}'
    if not os.path.exists(file_path):
        print('aborting')
        abort(404)

    return render_template(VIDEO_TEMPLATE_NAME, filename=filename)


# Next endpoints are support. Not for user
@model_page.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')

    if file is None:
        flash('>>> No file')
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file.save(f'{app.static_folder}/{filename}')

    app_root = app.config['APPLICATION_ROOT']

    return redirect(f'{app_root}/demo/{filename}')


@model_page.route('/demo/<filename>/video_feed')
def video_feed(filename):
    return Response(
        model_service.demo(f'{app.static_folder}/{filename}'),
        mimetype=VIDEO_FEED_MIMETYPE
    )


@model_page.errorhandler(404)
def not_found_error(e):
    app.logger.info(str(e))
    return render_template(ERROR_TEMPLATE_NAME, text=str(e))

import os

from flask import request, flash, redirect, jsonify, url_for, Blueprint, render_template, Response
from werkzeug.utils import secure_filename
import cv2

from config import VIDEO_TEMPLATE_NAME, UPLOAD_TEMPLATE_NAME
from controller.app import app
from service.model_service import ModelService

model_page = Blueprint('model_page', __name__)
model_service = ModelService(app.logger)


@model_page.route('/index')
def index():
    return render_template(UPLOAD_TEMPLATE_NAME)


@model_page.route('/demo/<filename>')
def demo(filename):
    print(f'>>> {filename}')
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

    filename = os.path.splitext(filename)[0]
    app_root = app.config['APPLICATION_ROOT']
    return redirect(f'{app_root}/demo/{filename}')


@model_page.route('/demo/<filename>/video_feed')
def video_feed(filename):
    print(f'video_feed: {filename}')
    return Response(
        model_service.demo('C:/Users/eduard/Desktop/clodding_train.mp4'),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

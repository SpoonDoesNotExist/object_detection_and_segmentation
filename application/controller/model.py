from flask import request, flash, redirect, jsonify, url_for, Blueprint, render_template
from werkzeug.utils import secure_filename

from config import VIDEO_TEMPLATE_NAME
from controller.app import app
from service.model_service import ModelService

model_page = Blueprint('model_page', __name__)
model_service = ModelService(app.logger)


@model_page.route('/demo', methods=['POST'])
def demo():
    file = request.files.get('file')
    if file is None:
        flash('No file')
        return redirect(request.url)

    flash('Processing video...')

    filename = secure_filename(file.filename)
    file.save(app.config['UPLOAD_FOLDER'] / filename)
    return jsonify(success=True)


@model_page.route('/display/<filename>')
def display_video(filename):
    video_url = url_for('static', filename=filename + '.mp4')
    return render_template(VIDEO_TEMPLATE_NAME, video_url=video_url)

# http://192.168.0.106:5000/api/display/clodding_train

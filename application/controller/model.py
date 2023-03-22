from .app import app
from application.service.model_service import ModelService

model_service = ModelService(app)


@app.route('/demo', methods=['POST'])
def demo():
    raise NotImplementedError()


@app.route('/evaluate', methods=['POST'])
def predict():
    raise NotImplementedError()

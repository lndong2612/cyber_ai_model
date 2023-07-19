import logging

from flask_cors import CORS
from config import settings
from flask import Flask, jsonify, abort
from app.api import route_model_ai
from flasgger import Swagger
from app.database import executor


def create_app():
    app = Flask(__name__)
    app.config.from_object("config.settings")
    app.register_blueprint(route_model_ai)

    logging.basicConfig(level=logging.INFO,
                        format=u"%(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    CORS(app, resources={r"/*": {"origins": "*"}})
    executor.init_app(app)
    config_swagger = {
        "specs": [
            {
                "endpoint": 'apispec_1',
                "route": '/{}/apispec_1.json'.format(settings.API_V1_STR),
                "rule_filter": lambda rule: True,  # all in
                "model_filter": lambda tag: True,  # all in
            }
        ],
        "static_url_path": "/{}/flasgger_static".format(settings.API_V1_STR),
        "specs_route": "/{}/apidocs/".format(settings.API_V1_STR)
    }
    swagger = Swagger(app, config=config_swagger, merge=True, template_file='swagger.yaml')

    @app.errorhandler(400)
    def handle_400(e):
        return jsonify({'message': str(e)}), 400

    @app.errorhandler(404)
    def handle_404(e):
        return jsonify({'message': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logging.error(str(e))
        return jsonify({'message': 'Something went wrong, we are working on it'}), 500

    print(settings.MODEL_FOLDER)
    return app


app = create_app()

if __name__ == "__main__":
    host = settings.HOST
    port = settings.PORT
    app.run(host=host, port=port, debug=settings.DEBUG, threaded=True)

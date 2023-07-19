from flask import Blueprint
from config import settings
from .model_api import control_model
from .deploy_api import control_deploy
from .noise_api import control_noise

route_model_ai = Blueprint("route_model_ai", __name__, url_prefix=settings.API_V1_STR)

route_model_ai.register_blueprint(control_model)
route_model_ai.register_blueprint(control_deploy)
route_model_ai.register_blueprint(control_noise)
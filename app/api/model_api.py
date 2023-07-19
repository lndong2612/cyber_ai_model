import os
import logging

from config import settings
from app.database import Model
from flask import Blueprint, jsonify, request
from app.auth.jwt_decode import token_required
from app.neural_model.list_model import all_model
from flasgger import swag_from

control_model = Blueprint('control_model', __name__)

logger = logging.getLogger(__name__)


@control_model.route("/train/<id_model>", methods=['GET'])
@token_required
def train_model(id_model):
    data_update = {
        'model_status': 'training'
    }
    model_train = None
    model_update_db = Model.get(id_model)
    if model_update_db:
        model_architecture = model_update_db.architecture
        model_train = all_model.get(model_architecture)
    if model_train:
        model_update_db = Model.update(id_model, data_update)
        model_train().train_model(id_model)
        # executor.submit(model.train_model, id_model)
    else:
        return jsonify(sucess=False, message="Chưa hỗ trợ mô hình này"), 400
    return jsonify(success=True)


@control_model.route("/action/<id_model>", methods=['PUT'])
@token_required
def action_model(id_model):
    action = request.json.get('action')
    data_update = {
        'model_status': 'training'
    }
    if action not in ['active', 'inactive', 'stop']:
        return jsonify(success=False, message='Không hỗ trợ thao tác này'), 400
    model_update_db = Model.get(id_model)
    if model_update_db:
        data_update = {
            'model_status': action
        }
        Model.update(id_model, data_update)
    else:
        return jsonify(sucess=False, message="Không tìm thấy mô hình này"), 400
    return jsonify(success=True)

@control_model.route("/upload_weight", methods=['POST'])
def upload_weight():
    file = request.files.get('file')
    id_model = request.form.get('id_model')
    if id_model:
        path_save = os.path.join(settings.MODEL_FOLDER, '{}/train_model/weights'.format(id_model), file.filename)
    else:
        path_save = os.path.join(settings.MODEL_WEIGHT_INIT, file.filename)
    if not os.path.exists( os.path.dirname(path_save)):
        os.makedirs( os.path.dirname(path_save))
    if not os.path.exists(path_save):
        file.save(path_save)
    return jsonify(success=True)
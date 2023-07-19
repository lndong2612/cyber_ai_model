import os
import cv2
import logging
import numpy as np

from config import settings
from constant import StatusModel
from cachetools import LRUCache
from app.database.model_db import Model
from flask import Blueprint, jsonify, request, make_response
from app.neural_model.list_model import all_model
from app.utils.get_model_deploy import CachedModelRepository, ModelRepository
from app.auth.jwt_decode import token_required

control_deploy = Blueprint('control_deploy', __name__)

logger = logging.getLogger(__name__)

model_cache_repository = CachedModelRepository(
    LRUCache(maxsize=settings.MAX_MODEL)
)


@control_deploy.route("/deploy/<id_model>", methods=['GET'])
@token_required
def deploy_model(id_model):
    data_update = {
        'model_status': StatusModel.ACTIVE
    }
    model_update_db = Model.get(id_model)
    if model_update_db:
        if model_update_db.model_status != StatusModel.TRAINED:
            return jsonify(success=False, message="Mô hình chưa được huấn luyện xong"), 400
        data_update = {
            'model_status': 'active'
        }
        Model.update(id_model, data_update)
    else:
        return jsonify(sucess=False, message="Không tìm thấy mô hình này"), 400
    return jsonify(success=True)


@control_deploy.route("/predict/<id_model>", methods=['POST'])
def predict_model(id_model):
    file = request.files.get('file')
    npimg = np.fromfile(file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    model_repo = ModelRepository()
    model_db = Model.get(id_model)
    model_arch = model_db.architecture
    model_cache_predict = model_cache_repository.get_model(id_model, model_arch, model_repo)
    model = all_model.get(model_arch)
    if model:
        result = model().predict(id_model, image, model_cache_predict)
    return result


@control_deploy.route("/test/<id_model>", methods=['POST'])
def test_model(id_model):
    file = request.files.get('file')
    npimg = np.fromfile(file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    model_repo = ModelRepository()
    model_arch = 'mobilenet'
    model_cache_predict = model_cache_repository.get_model(id_model, model_arch, model_repo)
    model = all_model.get(model_arch)
    if model:
        classified = model().predict(id_model, image, model_cache_predict)
        img = model().plot_bbox_image(image, classified)
        if not os.path.exists(settings.TEST_FOLDER):
            os.makedirs(settings.TEST_FOLDER)
        path_save_image = os.path.join(settings.TEST_FOLDER, 'test.png')
        cv2.imwrite(path_save_image, img)
    return jsonify(classified=classified, img_url="/model/resources/test.png", success=True)


@control_deploy.route("/resources/test.png", methods=['GET'])
def get_image():
    path_image = os.path.join(settings.TEST_FOLDER, 'test.png')
    if not os.path.exists(path_image):
        return jsonify(message="Không có ảnh"), 400
    image = cv2.imread(path_image)
    if image.shape[1] > 3500:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45]
    elif image.shape[1] > 3000:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 55]
    elif image.shape[1] > 2000:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    elif image.shape[1] > 1000:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    else:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    image_bytes = cv2.imencode(".jpg", image, encode_param)[1].tobytes()
    response = make_response(image_bytes)
    response.headers.set("Content-Type", "image/jpeg")
    response.headers.set("Content-Disposition",
                         "attachment", filename='test.jpg')
    return response

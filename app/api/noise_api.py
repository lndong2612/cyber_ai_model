import logging

from flask import Blueprint, jsonify, request


control_noise = Blueprint('control_noise', __name__)

logger = logging.getLogger(__name__)

@control_noise.route('/noise', methods=['POST'])
def noise_model():
    id_dataset = request.json.get('id_dataset')
    id_dataset_noise = request.json.get('id_dataset_noise')
    id_model = request.json.get('id_model')
    attack_method = request.json.get('attack_method')
    threshold = request.json.get('threshold')
    return jsonify(success=True), 200
import os
import cv2
import torch

from config import settings
from .train import train_mobilefacenet
from app.database.model_db import Model
from app.neural_model.base_model import BaseModel
from app.utils.thread_training import Training
from app.neural_model.mobile_facenet.utils import create_model_mobileface_retrain
from app.neural_model.mobile_facenet.face_detect import predict_face
from app.neural_model.mobile_facenet.utils import plot_bbox

class MobileNet(BaseModel):
    name_model = 'mobilenet'

    def get_model(self, id_model):
        # model_db = Model.get(id_model)
        label_list = ['0', '1']

        base_path = '{}/train_model/weights/best.pt'.format(id_model)
        model_weight = os.path.join(settings.MODEL_FOLDER, base_path)

        num_class = len(label_list)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = create_model_mobileface_retrain(device=device, num_class=num_class, model_weight=model_weight)
        return model

    def train_model(self, id_model):
        model_db = Model.get(id_model)
        # config model
        learning_rate = model_db.config.get('learning_rate') or 0.001
        epochs = model_db.epochs or 20
        batch_size = model_db.batch_size or 8
        labels_list = model_db.labels
        labels_list = [str(label) for label in labels_list]
        weight_init = os.path.join(settings.MODEL_WEIGHT_INIT, 'model_mobilefacenet.pth')
        # create thread train
        thread_train = Training()
        thread_train.start_train(
            id_model, train_mobilefacenet,
            [id_model, learning_rate, epochs, batch_size, labels_list, weight_init])

    def predict(self, id_model, input, model_cache):
        # model_db = Model.get(id_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        label_list = ['Long', 'Vu']
        classified = predict_face(model_cache, input, settings.THRESHOLD_FACE, device, label_list)
        return classified

    def plot_bbox_image(self, image, bboxes):
        img = plot_bbox(image, bboxes)
        return img

import os
from config import settings
from app.neural_model.base_model import BaseModel
from app.neural_model.attention_ocr.predict_lp import predict_lp
from app.neural_model.attention_ocr.load_model import load_aocr_model
import app.neural_model.attention_ocr.core.utils as utils

class AOCR(BaseModel):
    name_model = 'attention_ocr'

    def get_model(self, id_model):
        base_path = '{}/train_model/weights/aocr_model'.format(id_model)
        model_weight = os.path.join(settings.MODEL_FOLDER, base_path)
        model = load_aocr_model(model_weight)

        return model
    
    # def train_model(self, id_model):
    #     model_db = Model.get(id_model)
    #     # config model
    #     learning_rate = model_db.config.get('learning_rate') or 0.001
    #     epochs = model_db.epochs or 20
    #     batch_size = model_db.batch_size or 8
    #     labels_list = model_db.labels
    #     labels_list = [str(label) for label in labels_list]
    #     weight_init = os.path.join(settings.MODEL_WEIGHT_INIT, 'model_mobilefacenet.pth')
    #     # create thread train
    #     thread_train = Training()
    #     thread_train.start_train(
    #         id_model, train_mobilefacenet,
    #         [id_model, learning_rate, epochs, batch_size, labels_list, weight_init])        

    def predict(self, input, model_cache):
        classified = predict_lp(model_cache, input)
        return classified
    
    def plot_bbox_image(self, image, bboxes):
        img_bbox = utils.draw_bbox(image, bboxes)
        return img_bbox

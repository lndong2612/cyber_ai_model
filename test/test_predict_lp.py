import os
import sys
import cv2
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WORKING_DIR, '../'))
from config import settings

from app.neural_model.attention_ocr.predict_lp import predict_lp
from app.neural_model.attention_ocr.load_model import load_aocr_model
# plate_45400_15122021-113506.jpg - 1 plate
# plate_125606_15122021-131916.jpg - 2 plates
img_full = './test/image_test/cam2.jpg'
id_model = 18072023154736
base_path = '{}/train_model/weights/aocr_model'.format(id_model)
model_weight = os.path.join(settings.MODEL_FOLDER, base_path)
model = load_aocr_model(model_weight)
classified = predict_lp(model, img_full)
print(classified)
print("Done")


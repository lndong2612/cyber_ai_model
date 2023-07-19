
import os
import sys
import cv2

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WORKING_DIR, '../'))

from app.neural_model.mobile_facenet.utils import create_model_mobileface_retrain
from app.neural_model.mobile_facenet.face_detect import predict_face

device = 'cpu'
num_class = 2
model_weight = '/home/tranvien98/1Thinklabs/Year_2023/Cyber/CyberSecurityBE/public/files/models/646adf7b1fddc853312c8a3e/train_model/weights/best.pt'

model = create_model_mobileface_retrain(device=device, num_class=num_class, model_weight=model_weight)

image = cv2.imread('./test/image_test/vu.jpg')

label_list = ['Long', 'Vu']
classified = predict_face(model, image, 0.45, device, label_list)
print(classified)

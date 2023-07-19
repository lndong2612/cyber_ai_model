from app.neural_model.mobile_facenet.mobile_model import MobileNet
from app.neural_model.yolov5.yolov5_model import YOLOV5
from app.neural_model.attention_ocr.aocr_model import AOCR
from app.neural_model.wpodnet.wpod_model import WPOD

all_model = {
    'mobileface': MobileNet,
    'yolov5': YOLOV5,
    'attention_ocr': AOCR,
    'wpod_net': WPOD
}

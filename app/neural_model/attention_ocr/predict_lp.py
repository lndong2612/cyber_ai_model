import cv2
from app.neural_model.attention_ocr.core.config import cfg
import app.neural_model.attention_ocr.core.utils as utils
from app.neural_model.attention_ocr.detection_lp import detect_plate

def predict_lp(model, full_image):
    classified = []    

    size = 416
    iou = 0.45
    score = 0.50

    # load image
    original_image = cv2.imread(full_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # get bounding box from detection 
    pred_bbox = detect_plate(original_image, size, iou, score)   
    out_boxes, out_scores, out_classes, num_boxes = pred_bbox
    if out_boxes is not None:
        for num_box in range(num_boxes):
            # get crop plate
            img_crop, xmin, ymin, xmax, ymax = utils.crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), out_boxes, num_box, out_classes)

            # get predict
            results_process = utils.predict_crop(model, img_crop)
            doc = {'xmin': int(xmin), 
                   'ymin': int(ymin), 
                   'xmax': int(xmax), 
                   'ymax': int(ymax), 
                   'score': '',
                   'polygons': [[]], 
                   'label': results_process
                   }
            classified.append(doc)

    return classified

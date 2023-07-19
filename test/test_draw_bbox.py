import os
import sys
import cv2
import numpy as np
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WORKING_DIR, '../'))
import app.neural_model.attention_ocr.core.utils as utils
classifier = [{'xmin': 175, 'ymin': 907, 'xmax': 274, 'ymax': 970, 'score': '', 'polygons': [[]], 'label': '29-G1 26645'}, {'xmin': 1417, 'ymin': 404, 'xmax': 1474, 'ymax': 445, 'score': '', 'polygons': [[]], 'label': '29-M1 07457'}]
# classifier = [{'xmin': 1023, 'ymin': 578, 'xmax': 1090, 'ymax': 633, 'score': '', 'polygons': [[]], 'label': '54-P4 0545'}]
img_full = './test/image_test/cam2.jpg'
img = cv2.imread(img_full)
img_bbox = utils.draw_bbox(img, classifier)

cv2.imwrite('bbox.jpg', img_bbox)

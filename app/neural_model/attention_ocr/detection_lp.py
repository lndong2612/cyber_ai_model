import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import app.neural_model.attention_ocr.core.utils as utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def detect_plate(original_image, input_size, iou, score):
    weight_path = './app/neural_model/attention_ocr/checkpoints/plate-416'
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # load model
    saved_model_loaded = tf.saved_model.load(weight_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # loop through images in list and run Yolov4 model on each
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class = 50,
        max_total_size = 50,
        iou_threshold = iou,
        score_threshold = score
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    return pred_bbox

import cv2
from facenet_pytorch import MTCNN
from app.neural_model.mobile_facenet.utils import extract_face
from app.neural_model.mobile_facenet.test import predict_image


def predict_face(model, image, conf_thres, device, labels_object):
    classified = []
    model.eval()
    image0 = image.copy()
    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)
    boxes, _, _ = mtcnn.detect(image0, landmarks=True)
    if boxes is not None:
        for box in boxes:
            bbox = list(map(int, box.tolist()))
            image0 = cv2.rectangle(
                image0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
            if abs(bbox[3] - bbox[2]) > 0 and abs(bbox[1]-bbox[0]) > 0:
                face = extract_face(bbox, image0)
                label, score = predict_image(
                    model, labels_object, face, device)
                if score > conf_thres:
                    doc = {'xmin': int(bbox[0]), 'ymin': int(bbox[1]), 'xmax': int(bbox[2]), 'ymax': int(
                        bbox[3]), 'score': float(score), 'polygons': [[]], 'label': label}
                    classified.append(doc)
    return classified

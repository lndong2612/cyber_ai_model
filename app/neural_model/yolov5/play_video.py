import cv2
import torch
import time
import numpy as np
from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression

def load_model(weights):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights, device)  # load FP32 model
    return model


def predict_one_box(x, im):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    return c1[0], c1[1], c2[0], c2[1]

def predict_image(model, im0s, imgsz, conf_thres, device):
    """
    model: mô hình dự đoán 
    i0s: ảnh cần dự đoán
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    img = letterbox(im0s, imgsz)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    # Inference
    pred = model(img, augment=False, visualize=False)[0]
    # NMS
    pred = non_max_suppression(
        pred, conf_thres, 0.45, None, False, max_det=1000)
    classified = []
    for _, det in enumerate(pred):  # detections per image
        im0 = im0s.copy()
        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if float(conf) > conf_thres:
                    label = names[c]
                    nhan = predict_one_box(xyxy, im0)
                    doc = {'xmin': int(nhan[0]), 'ymin': int(nhan[1]), 'xmax': int(nhan[2]), 'ymax': int(
                        nhan[3]), 'confidence': float(conf), 'polygons': [[]], 'label': label, "fault": False}
                    classified.append(doc)
    return classified


url_video = 'data/video/video_2023-06-05_15-58-28.mp4'
model_load = 'weights/yolov5m-face.pt'
model = load_model(model_load)
# check device

cap = cv2.VideoCapture(url_video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    bboxes = predict_image(model, frame, 640, 0.6, 'cpu')
    print(bboxes)
    h, w, c = frame.shape
    width_resize = 400
    height_resize = round(width_resize*h/w)
    frame = cv2.resize(frame, (width_resize, height_resize), cv2.INTER_AREA)
    time.sleep(0.05)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
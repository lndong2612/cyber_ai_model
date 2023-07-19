import torch
import os
import cv2
import colorsys
import unicodedata
import numpy as np
import torch.nn as nn

from app.utils.label import convet_list_label_id
from facenet_pytorch import MTCNN
from app.neural_model.mobile_facenet.helper import MobileFaceNet
from app.database.model_db import Model as db_model


class SaveBestModel:
    """
    Lưu lại mô hình khi val_loss < train_loss
    """

    def __init__(self, path_save, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.path_save = path_save

    def __call__(self, current_valid_loss, epoch, model):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save(model.state_dict(), self.path_save)


def create_model_mobileface(device: str, num_class: int, model_weight: str = None):
    """
    Hàm tạo model resnet 
    Parameters
        device: thiết bị sử dụng để nhận diện mô hình đào tạo
    """
    # try:
    model = MobileFaceNet(512)
    model.load_state_dict(torch.load(model_weight, map_location=device))
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    model.bn = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_class),
    )
    model.to(device)
    return model
    # except Exception as e:
    #     print(e)


def create_model_mobileface_retrain(device: str, num_class: int, model_weight: str = None):
    """
    Hàm tạo model resnet 
    Parameters
        device: thiết bị sử dụng để nhận diện mô hình đào tạo
    """
    # try:
    model = MobileFaceNet(512)
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    model.bn = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_class),
    )
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model.to(device)
    return model


def select_device_gpu(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace(
        'cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        # force torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(
        ), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        device = device if device else '0'
        # print(devices)
        torch.device("cuda:" + str(device) if cuda else 'cpu')
    else:
        s += 'CPU\n'
    return torch.device("cuda:0" if cuda else 'cpu')


def extract_face(box, img, margin=20):
    face_size = 112
    img = img[box[1]:box[3], box[0]:box[2], :]
    face = cv2.resize(img, (face_size, face_size),
                      interpolation=cv2.INTER_AREA)
    # face = Image.fromarray(face)
    return face


def max_area_box(boxes):
    area = [int(bbox[2]-bbox[0]) * int(bbox[3]-bbox[1]) for bbox in boxes]
    # print(area)
    res = max(enumerate(area), key=lambda x: x[1])
    return res[0]


def cut_image_max(img, device):
    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)
    boxes, _, _ = mtcnn.detect(img, landmarks=True)
    count = 10
    if boxes is not None:
        index_max = max_area_box(boxes)
        box = boxes[index_max]
        count += 1
        bbox = list(map(int, box.tolist()))
        # img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(img.shape[1], bbox[2])
        bbox[3] = min(img.shape[0], bbox[3])
        if bbox[3] - bbox[1] > 0 and bbox[2]-bbox[0] > 0:
            face = extract_face(bbox, img)
        # print(face.shape, bbox)
        return face, bbox
    return None, None


def predict_one_box(x, im):
    # Plots one bounding box on image 'im' using OpenCV
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    return c1[0], c1[1], c2[0], c2[1]


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return text.replace("đ", "d").replace("Đ", "D")


def plot_bbox(image, bounding_box):
    width, height = image.shape[:2]
    catategory = []
    for bbox in bounding_box:
        catategory.append(bbox["label"])
    num_classes = 30
    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(30)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255),
            int(x[1] * 255), int(x[2] * 255)), colors)
    )
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bbox = bounding_box
    for box in bbox:
        left = int(float(box["xmin"]))
        top = int(float(box["ymin"]))
        right = int(float(box["xmax"]))
        bottom = int(float(box["ymax"]))
        cl = box["label"]
        bbox_color = colors[list(catategory).index(cl)]
        bbox_thick = 1 if min(width, height) < 800 else 3
        cv2.rectangle(image, (left, top), (right, bottom),
                      bbox_color, bbox_thick)
        bbox_mess = "%s" % remove_accents(cl)
        t_size = cv2.getTextSize(bbox_mess, 0, bbox_thick, thickness=1)[0]
        cv2.rectangle(
            image, (left, top), (left + t_size[0], top - t_size[1] - 3), (255, 255, 255), -1
        )
        cv2.putText(
            image,
            bbox_mess,
            (left, top - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            bbox_thick,
            (0, 0, 0),
            bbox_thick,
            lineType=cv2.LINE_AA,
        )

        mask[top:bottom, left:right] = 255
        mask[top - t_size[1] - 3:top, left: left+t_size[0]] = 255


    foreground = cv2.bitwise_or(image, image, mask=mask)
    alpha = 0.55
    blended_image = cv2.addWeighted(image, alpha, foreground, 1-alpha, 0, image)
    return blended_image


def update_evalution_to_db(id_model, label_list, test_acc, cf_matrix):
    results_per_class = {}
    confusion_matrix = {}

    label_list = convet_list_label_id(label_list)
    nb_classes = len(label_list)

    b = np.where(np.isnan(cf_matrix), 0, cf_matrix)
    cf_matrix = torch.from_numpy(b)
    precision = cf_matrix.diagonal()/(cf_matrix.sum(1) + 1e-6)
    recall = cf_matrix.diagonal()/(cf_matrix.sum(0) + 1e-6)
    for i in range(nb_classes):
        if cf_matrix[i].sum() == 0:
            cf_matrix[i] = 0
        cf_matrix[i] = cf_matrix[i] / (cf_matrix[i].sum() + 1e-6)
    for key in label_list:
        results_per_class[key] = {
            "precision": round(precision[label_list.index(key)].item(), 2),
            "recall": round(recall[label_list.index(key)].item(), 2)
        }
    confusion_matrix = {
        "label": list(label_list),
        "data": cf_matrix.cpu().detach().numpy().tolist()
    }
    evalution_res = {
        "precision": round(precision.mean().item(), 2),
        "recall": round(recall.mean().item(), 2),
        "accuracy": round(test_acc, 2),
        "maps": (precision.mean().item()+recall.mean().item())/2, #công thức sai 
        "confusion_matrix": confusion_matrix,
        "results_per_class": results_per_class
    }
    # print("Evalution model",evalution_res)
    print("save model")
    db_model.update(id_model=id_model, data_update={
        'evalution': evalution_res,
        'model_status': 'trained'
    })



class UpdateLogDB():
    def __init__(self, id_model, epochs):
        self.train_log = []
        self.val_log = []
        self.id_model = id_model
        self.epochs = epochs

    def update_db(self, train_loss, valid_loss, valid_acc, epoch, estimate_time):
        self.train_log.append(
            {
                "epoch": epoch + 1,
                "value": round(train_loss, 3)
            }
        )
        self.val_log.append(
            {
                "epoch": epoch + 1,
                "value": round(valid_loss, 3)
            }
        )
        log = {
            "epoch": epoch+1,
            "total_epoch": self.epochs,
            "estimate_time": round(estimate_time, 3),
            "train_log": self.train_log,
            "val_log": self.val_log
        }
        evalution_res = {
            "precision": valid_acc
        }
        db_model.update(id_model=self.id_model, data_update={
            'log': log,
            'evalution': evalution_res
        })

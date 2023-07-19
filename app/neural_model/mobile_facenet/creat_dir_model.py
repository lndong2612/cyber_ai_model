import os
import cv2
import logging
import torch

from config import settings
from app.neural_model.mobile_facenet.utils import cut_image_max
from sklearn.model_selection import train_test_split
from app.database import Model, Dataset, Image, Label


def create_data_train_mobileface(id_model: str):
    """
    Xác định vùng chứa khuôn mặt, tạo dữ liệu train test
    """
    model_db = Model.get(id_model)
    assert model_db is not None, logging.error("Mô hình không tồn tại")
    if not model_db:
        logging.error("Mô hình không tồn tại dòng 17 file mobile_facênt/create_dir_model.")
        return
    # labels = list(model.config.get('labels_object').keys())
    trainset_rate = model_db.split_train or 0.8
    trainset, testset = [], []
    labels_list = model_db.labels
    labels_list = [str(label) for label in labels_list]
    images_and_labels = get_images_and_labels_in_dataset(model_db.dataset, labels_list)
    for key in images_and_labels.keys():
        train, test = train_test_split(images_and_labels[key], train_size=trainset_rate, random_state=42)
        trainset.extend(train)
        testset.extend(test)
    return trainset, testset


def get_images_and_labels_in_dataset(id_dataset, labels_list):
    images_and_labels = {}
    dataset_db = Dataset.get(id_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataset_db:
        for image_id in dataset_db.images:
            image_db = Image.get(image_id)
            image_url = image_db.img_path.replace(settings.RESOURCES_NODE, settings.RESOURCES)
            if os.path.exists(image_url):
                img = cv2.imread(image_url)
                face, bbox = cut_image_max(img, device)
                if face is not None and bbox is not None:
                    for label_db in image_db.labels:
                        if not label_db:
                            continue
                        if str(label_db['labelId']) not in labels_list:
                            continue
                        label_id = str(label_db['labelId'])
                        label_index = labels_list.index(label_id)
                        if not images_and_labels.get(label_id):
                            images_and_labels[label_id] = []
                        images_and_labels[label_id].append({
                            'label': label_index,
                            'image_url': image_url,
                            'xmin': bbox[0],
                            'ymin': bbox[1],
                            'xmax': bbox[2],
                            'ymax': bbox[3]
                        })
    return images_and_labels

from datetime import datetime, timezone
import time
from bson import ObjectId
from uuid import uuid4
from app.config import settings
from app.database.database_synch import db_synch
import pandas as pd
import cv2
import torch
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from app.model_ai.mobile_facenet.utils import create_model_mobileface_retrain
import numpy as np
import torch.optim as optim
import os
import sys
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WORKING_DIR, "../"))


def make_unique(string):
    ident = uuid4().__str__()[:8]
    return f"{ident}-{string}"


def creat_noise(attack, classifier, path_data, labels_object, device, id_datanoise, db):
    total = 0
    correct = 0
    num_class = len(labels_object.keys())
    cf_matrix = torch.zeros(num_class, num_class)
    data = pd.read_csv(path_data)
    image_array = []
    for idx in range(len(data)):
        image_url = data.iloc[idx]['image_url']
        imageAttackID = data.iloc[idx]['imageAttackID']
        xmin, xmax, ymin, ymax = data.iloc[idx]['xmin'], data.iloc[idx]['xmax'], data.iloc[idx]['ymin'], data.iloc[idx]['ymax']
        origin_path = data.iloc[idx]['origin_path']
        origin_image = cv2.imread(origin_path)
        basename = os.path.basename(image_url)
        inputs = cv2.imread(image_url)
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        width = xmax-xmin
        height = ymax-ymin
        print(width, height)
        inputs = cv2.resize(inputs, (112, 112))
        pre_img = np.stack([inputs], axis=0).astype(np.float32)
        pre_img = np.transpose(pre_img, (0, 3, 1, 2))
        labels = int(data.iloc[idx]['labels'])
        x_test_adv = attack.generate(x=pre_img)
        outputs = classifier.predict(x_test_adv)
        idxs = int(time.time() * 1000)
        extpath = basename.split(".")[-1]
        new_imgname = str(idxs) + '.' + extpath
        now = datetime.now()
        path_dir = os.path.join(settings.RESOURCES, str(now.year), str(now.month), str(now.day))
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        img_path = os.path.join(path_dir, new_imgname)
        if os.path.exists(img_path):
            new_imgname = make_unique(new_imgname)
            img_path = os.path.join(path_dir, new_imgname)
        print(img_path)
        x_test_adv_p = np.transpose(x_test_adv, (0, 2, 3, 1)).astype(np.float32)
        face = cv2.cvtColor(cv2.resize(x_test_adv_p[0], (width, height)), cv2.COLOR_BGR2RGB)
        origin_image[ymin:ymax, xmin:xmax] = face
        cv2.imwrite(img_path, origin_image)
        total += 1
        predictions = np.argmax(outputs, axis=1)
        correct += np.sum(predictions == labels)
        tmp_labels = {
            "labelId": ObjectId(list(labels_object.values())[predictions[0]])
        }
        img_dict = {
            "is_deleted": False,
            "labels": [tmp_labels],
            "img_name": new_imgname,
            "img_originalname": basename,
            "img_desc": "attack image",
            "img_uri": settings.DOMAIN + img_path.replace(settings.RESOURCES, '/resources/images'),
            "img_path": img_path.replace(settings.RESOURCES, '/resources/images'),
            "datanoiseId": ObjectId(id_datanoise),
            'imageAttackID': ObjectId(imageAttackID),
            "createdAt": datetime.now().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "updatedAt": datetime.now().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        }
        try:
            result_update = db_synch.add_image(img_dict)
            db_synch.update_img_datanoise(id_datanoise, {"images": ObjectId(result_update.inserted_id)})
        except:
            print(img_path)
    accuracy = (correct/total)
    return accuracy


def attack_mobile(path_model, eps, labels_object, device, id_datanoise):
    num_class = len(labels_object.keys())
    db_synch.connect_to_database(path=settings.MONGO_DATABASE_URI)
    path_train = os.path.join(path_model, 'data', 'train.csv')
    path_val = os.path.join(path_model, 'data', 'val.csv')
    path_test = os.path.join(path_model, 'data', 'test.csv')
    path_weight = os.path.join(path_model, 'train_model', 'weights', 'best.pt')
    model = create_model_mobileface_retrain("cpu", num_class, path_weight)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 255),
        loss=loss_criterion,
        optimizer=optimizer,
        input_shape=(3, 112, 112),
        nb_classes=num_class,
    )

    attack = FastGradientMethod(estimator=classifier, eps=eps)
    _ = creat_noise(attack=attack, classifier=classifier, path_data=path_train, labels_object=labels_object,
                    device=device, id_datanoise=id_datanoise, db=db_synch)
    _ = creat_noise(attack=attack, classifier=classifier, path_data=path_val, labels_object=labels_object,
                    device=device, id_datanoise=id_datanoise, db=db_synch)
    accuracy_test = creat_noise(attack=attack, classifier=classifier, path_data=path_test, labels_object=labels_object,
                                device=device, id_datanoise=id_datanoise, db=db_synch)
    res = {"accuracy": accuracy_test}
    db_synch.update_datanoise(id_datanoise=id_datanoise, datanoise={"datanoise_status": "complete", "evalution": res})
    db_synch.connect_to_database(path=settings.MONGO_DATABASE_URI)

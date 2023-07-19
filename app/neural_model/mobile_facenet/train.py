import os
import time
import threading
import torch
import numpy as np
import torch.nn as nn

from typing import List
from tqdm import tqdm
from pathlib import Path
from config import settings
from torch.utils.data import DataLoader
from torchvision import transforms
from app.database.model_db import Model as db_model
from .creat_dir_model import create_data_train_mobileface
from app.neural_model.mobile_facenet.dataload import FaceDataset
from app.neural_model.mobile_facenet.utils import create_model_mobileface_retrain
from app.neural_model.mobile_facenet.utils import SaveBestModel, create_model_mobileface,\
    select_device_gpu, UpdateLogDB, update_evalution_to_db


def load_data(id_model, batch_size):
    image_transforms = {
        "train": transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(112, 112)),
            transforms.ToTensor(),
            lambda x: x*255
        ]),
        "valid": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(112, 112)),
            transforms.ToTensor(),
            lambda x: x*255
        ]),
        "test": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(112, 112)),
            transforms.ToTensor(),
            lambda x: x*255
            # transforms.Normalize(mean=(0,)*3, std=(255,)*3)
        ])
    }
    train_set, test_set = create_data_train_mobileface(id_model)
    dataset = {
        "train": FaceDataset(train_set, image_transforms["train"]),
        "test": FaceDataset(test_set, image_transforms["test"])
    }
    train_data = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    test_data = DataLoader(dataset["test"], batch_size=batch_size, shuffle=True)
    return train_data, test_data


def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_loss / counter
    epoch_acc = train_correct/(len(trainloader.dataset))
    return epoch_loss, epoch_acc


def valid_epoch(model, validloader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader), total=len(validloader)):
            counter += 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_correct += (preds == labels).sum().item()

    epoch_loss = valid_loss / counter
    epoch_acc = valid_correct/(len(validloader.dataset))
    return epoch_loss, epoch_acc


def evalution(model, testloader, criterion, nb_classes, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    counter = 0
    cf_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            test_correct += (preds == labels).sum().item()
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cf_matrix[t.long(), p.long()] += 1
    epoch_loss = test_loss / counter
    epoch_acc = test_correct/(len(testloader.dataset))
    return epoch_loss, epoch_acc, cf_matrix


def train(id_model, save_dir, learning_rate, epochs, batch_size, device, label_list, weight_init):

    device = select_device_gpu(device)
    save_dir = Path(save_dir)

    trainloader, testloader = load_data(id_model, batch_size)
    validloader = testloader
    path_weights = os.path.join(save_dir, "weights")
    path_save_model = os.path.join(path_weights, "best.pt")
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)
    save_best_model = SaveBestModel(path_save_model)
    update_log_db = UpdateLogDB(id_model, epochs)
    nb_classes = len(label_list)

    print("create_model")
    if weight_init.find('best.pt') != -1:
        model = create_model_mobileface_retrain(device, nb_classes, weight_init)
    else:
        model = create_model_mobileface(device, nb_classes, weight_init)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    t0 = int(time.time())

    # check to stop model
    t = threading.currentThread()
    for epoch in range(epochs):
        # epoch_start = time.time()
        model.train()
        if not getattr(t, "do_run", True):
            break
        # Loss and Accuracy within the epoch
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, loss_criterion, device)
        valid_loss, valid_acc = valid_epoch(model, validloader, loss_criterion, device)

        estimate_time = (time.time() - t0) * (epochs - epoch - 1) / (epoch+1)
        save_best_model(valid_loss, epoch, model)
        update_log_db.update_db(train_loss, valid_loss, valid_acc, epoch, estimate_time)
        print("Epochs: {} Train Loss: {} Train Acc: {} Valid Loss {} Valid Acc {} \n".format(
            epoch, round(train_loss, 3), round(train_acc, 3), round(valid_loss, 3), round(valid_acc, 3)), flush=True)

    _, test_acc, cf_matrix = evalution(model, testloader, loss_criterion, nb_classes, device)
    update_evalution_to_db(id_model, label_list, test_acc, cf_matrix)
    torch.cuda.empty_cache()


def train_mobilefacenet(
        id_model: str, learning_rate: float, epochs: int, batch_size: int, labels_list: List, weight_init: str):
    save_dir = os.path.join(settings.MODEL_FOLDER, str(id_model), 'train_model')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(id_model, save_dir, learning_rate, epochs, batch_size, device, labels_list, weight_init)

import torch
import pandas as pd
import cv2
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, data, transforms_data=None) -> None:
        self.data = data
        self.transforms_data = transforms_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]['image_url']
        label = self.data[idx]['label']
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
        if self.transforms_data:
            image = self.transforms_data(image)
        return image, label

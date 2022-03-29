import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, image_path: list, image_class: list, transform = None):
        self.image_path = image_path
        self.image_class = image_class
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item): #can add more rules to pick data
        img = Image.open(self.image_path[item])
        label = self.image_class[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, label





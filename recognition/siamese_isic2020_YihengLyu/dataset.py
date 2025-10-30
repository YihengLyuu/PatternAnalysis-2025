import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

# ---------------------- transforms ----------------------
def train_transforms(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def eval_transforms(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# ---------------------- 单图 ----------------------
class ISIC2020Singles(Dataset):
    def __init__(self, csv, images_dir, transform=None, ids_subset=None):
        df = pd.read_csv(csv)
        df["image_name"] = df["image_name"].astype(str)
        if ids_subset is not None:
            df = df[df["image_name"].isin(ids_subset)]
        self.samples = list(zip(df["image_name"].tolist(), df["target"].tolist()))
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        path_jpg = os.path.join(self.images_dir, f"{name}.jpg")
        path_png = os.path.join(self.images_dir, f"{name}.png")
        path = path_jpg if os.path.exists(path_jpg) else path_png
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(float(label)), name

# ---------------------- 成对数据 ----------------------
class ISIC2020Pairs(Dataset):
    def __init__(self, csv, images_dir, split="train", pairs_per_epoch=2000, img_size=192):
        df = pd.read_csv(csv)
        self.images_dir = images_dir
        self.img_size = img_size
        self.samples = df
        self.pairs_per_epoch = pairs_per_epoch
        self.transform = train_transforms(img_size) if split == "train" else eval_transforms(img_size)

        self.pos = df[df["target"] == 1]["image_name"].tolist()
        self.neg = df[df["target"] == 0]["image_name"].tolist()

    def __len__(self):
        return self.pairs_per_epoch

    def _load_img(self, name):
        path_jpg = os.path.join(self.images_dir, f"{name}.jpg")
        path_png = os.path.join(self.images_dir, f"{name}.png")
        path = path_jpg if os.path.exists(path_jpg) else path_png
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)

    def __getitem__(self, idx):
        same = random.random() > 0.5
        if same:
            cls = random.choice([self.pos, self.neg])
            if len(cls) >= 2:
                a, b = random.sample(cls, 2)
            else:
                a, b = cls[0], cls[0]
            y = 1.0
        else:
            a, b = random.choice(self.pos), random.choice(self.neg)
            y = 0.0
        return self._load_img(a), self._load_img(b), torch.tensor(y, dtype=torch.float32)

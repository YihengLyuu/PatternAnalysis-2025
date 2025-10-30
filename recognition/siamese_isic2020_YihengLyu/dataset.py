# dataset.py
# @edu:student-assignment
# Dataset and transform definitions for ISIC2020 Siamese training (with 'target' column support)

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import csv

# ===============================
# Image transformations
# ===============================
train_transforms = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===============================
# Siamese Pair Dataset
# ===============================
class ISIC2020Pairs(Dataset):
    """
    Produces pairs of (img1, img2, label)
    label = 1 if same class, 0 otherwise
    """

    def __init__(self, csv_records, images_dir, pairs_per_epoch=2000, transform=None):
        self.records = csv_records
        self.images_dir = images_dir
        self.pairs_per_epoch = pairs_per_epoch
        self.transform = transform

        # ---- Determine column indices ----
        header = self.records[0]
        if "image_name" in header and "target" in header:
            # If the first row is the header line
            self.records = self.records[1:]
            image_idx = header.index("image_name")
            label_idx = header.index("target")
        else:
            # Fallback (header already removed)
            image_idx = 0
            label_idx = -1

        # ---- Build class mapping ----
        self.class_to_imgs = {}
        for row in self.records:
            if not row or len(row) < 2:
                continue
            try:
                label = int(row[label_idx])
            except ValueError:
                continue
            image_id = row[image_idx].strip()
            if image_id == "":
                continue
            self.class_to_imgs.setdefault(label, []).append(image_id)

        self.classes = list(self.class_to_imgs.keys())

        if len(self.classes) == 0:
            raise ValueError("No valid samples found in CSV — check that 'target' column exists.")

    def __len__(self):
        return self.pairs_per_epoch

    def _load_img(self, image_id):
        path = os.path.join(self.images_dir, f"{image_id}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        same_class = random.random() < 0.5
        if same_class:
            c = random.choice(self.classes)
            img1_id, img2_id = random.sample(self.class_to_imgs[c], 2)
            label = 1
        else:
            c1, c2 = random.sample(self.classes, 2)
            img1_id = random.choice(self.class_to_imgs[c1])
            img2_id = random.choice(self.class_to_imgs[c2])
            label = 0

        img1 = self._load_img(img1_id)
        img2 = self._load_img(img2_id)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label


# ===============================
# Single-Image Dataset (for linear probe)
# ===============================
class ISIC2020Singles(Dataset):
    """
    Used in predict.py for single-image embedding extraction and evaluation.
    """
    def __init__(self, csv_records, images_dir, transform=None):
        self.records = csv_records
        self.images_dir = images_dir
        self.transform = transform

        # Auto-detect header if present
        if self.records[0][0] == "image_name":
            self.records = self.records[1:]
            self.image_idx = 0
            self.label_idx = -1
        else:
            self.image_idx = 0
            self.label_idx = -1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        image_id = row[self.image_idx]
        label = int(row[self.label_idx])
        path = os.path.join(self.images_dir, f"{image_id}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing image: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, image_id

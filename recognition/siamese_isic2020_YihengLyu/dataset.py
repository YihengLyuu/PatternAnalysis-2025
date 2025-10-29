import os
import random
import pandas as pd
from PIL import Image
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def default_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def eval_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class ISIC2020Pairs(Dataset):
    """
    动态在线采样“成对样本”(x1,x2,label_same)，label_same: 1=同类(相同target), 0=异类。
    假设图像路径: root_dir/<image_name>.jpg
    CSV 至少包含: image_name, target
    """
    def __init__(self,
                 csv_path: str,
                 root_dir: str,
                 transform=None,
                 pairs_per_epoch: int = 100000,
                 seed: int = 42):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform or default_transforms()
        self.rng = random.Random(seed)

        # 建索引：按类别分组
        self.by_cls = {}
        for cls, sub in self.df.groupby('target'):
            self.by_cls[int(cls)] = list(sub['image_name'])

        self.ids = list(self.df['image_name'])
        self.labels = dict(zip(self.df['image_name'], self.df['target']))
        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def _load_img(self, image_name: str):
        # 兼容 .jpg/.png
        for suf in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(self.root, image_name + suf)
            if os.path.exists(p):
                img = Image.open(p).convert('RGB')
                return img
        raise FileNotFoundError(f"Image file for {image_name} not found under {self.root}")

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 50% 同类，50% 异类
        same = self.rng.random() < 0.5

        # 先采第一个
        img1_name = self.rng.choice(self.ids)
        y1 = int(self.labels[img1_name])

        if same:
            img2_name = self.rng.choice(self.by_cls[y1])
            y = 1
        else:
            other_cls = 1 - y1  # 二分类
            # 若数据极不平衡，做个兜底：
            if other_cls not in self.by_cls or len(self.by_cls[other_cls]) == 0:
                other_cls = y1
                same = True
                y = 1
                img2_name = self.rng.choice(self.by_cls[other_cls])
            else:
                img2_name = self.rng.choice(self.by_cls[other_cls])
                y = 0

        img1 = self._load_img(img1_name)
        img2 = self._load_img(img2_name)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(float(y), dtype=torch.float32)

class ISIC2020Singles(Dataset):
    """
    用于评估/推理：单图 image -> label(0/1)
    """
    def __init__(self, csv_path: str, root_dir: str, transform=None, ids_subset=None):
        self.df = pd.read_csv(csv_path)
        if ids_subset is not None:
            self.df = self.df[self.df['image_name'].isin(ids_subset)].reset_index(drop=True)
        self.root = root_dir
        self.transform = transform or eval_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['image_name']
        label = int(row['target'])
        for suf in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(self.root, name + suf)
            if os.path.exists(p):
                img = Image.open(p).convert('RGB')
                img = self.transform(img)
                return img, label, name
        raise FileNotFoundError(f"Image file for {name} not found under {self.root}")

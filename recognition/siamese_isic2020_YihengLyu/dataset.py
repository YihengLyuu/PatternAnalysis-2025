import os
import random
from typing import Tuple, Optional, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def default_transforms(img_size: int = 192):
    # 更强数据增强，提升泛化
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def eval_transforms(img_size: int = 192):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class ISIC2020Pairs(Dataset):
    """
    在线采样 (x1, x2, same_label)，same_label: 1=同类，0=异类
    root_dir/<image_name>.jpg; CSV: image_name,target
    """
    def __init__(self,
                 csv_path: str,
                 root_dir: str,
                 transform=None,
                 pairs_per_epoch: int = 100000,
                 seed: int = 42):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.df["image_name"] = (self.df["image_name"]
                                 .astype(str)
                                 .str.replace(".jpg", "", regex=False)
                                 .str.replace(".png", "", regex=False))
        self.root = root_dir
        self.transform = transform or default_transforms()
        self.rng = random.Random(seed)

        self.by_cls = {}
        for cls, sub in self.df.groupby('target'):
            self.by_cls[int(cls)] = list(sub['image_name'])

        self.ids: List[str] = list(self.df['image_name'])
        self.labels = dict(zip(self.df['image_name'], self.df['target']))
        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def _load_img(self, image_name: str) -> Image.Image:
        for suf in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(self.root, image_name + suf)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f"Image file for {image_name} not found under {self.root}")

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        same = self.rng.random() < 0.5
        img1_name = self.rng.choice(self.ids)
        y1 = int(self.labels[img1_name])

        if same:
            img2_name = self.rng.choice(self.by_cls[y1])
            y = 1.0
        else:
            other_cls = 1 - y1
            if other_cls not in self.by_cls or len(self.by_cls[other_cls]) == 0:
                img2_name = self.rng.choice(self.by_cls[y1]); y = 1.0
            else:
                img2_name = self.rng.choice(self.by_cls[other_cls]); y = 0.0

        img1 = self._load_img(img1_name)
        img2 = self._load_img(img2_name)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(y, dtype=torch.float32)

class ISIC2020Singles(Dataset):
    """ 单图评估/推理 """
    def __init__(self, csv_path: str, root_dir: str, transform=None, ids_subset: Optional[List[str]] = None):
        self.df = pd.read_csv(csv_path)
        self.df["image_name"] = (self.df["image_name"]
                                 .astype(str)
                                 .str.replace(".jpg", "", regex=False)
                                 .str.replace(".png", "", regex=False))
        if ids_subset is not None:
            ids_subset = [str(x).replace(".jpg","").replace(".png","") for x in ids_subset]
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

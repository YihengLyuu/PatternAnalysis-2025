# @edu:student-assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 1) Siamese Network
# ============================================================
class SiameseNet(nn.Module):
    def __init__(self, backbone="resnet34", embed_dim=256):
        super().__init__()

        # ---- 选择 backbone ----
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=None)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights=None)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ---- 替换分类头 ----
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # ---- 新增线性层 ----
        self.fc = nn.Linear(in_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward_once(self, x):
        """提取单张图像的特征嵌入"""
        x = self.backbone(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2=None):
        """如果传入两张图，返回 (z1, z2)；否则返回单个嵌入"""
        if x2 is None:
            return self.forward_once(x1)
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2


# ============================================================
# 2) Contrastive Loss
# ============================================================
def contrastive_loss(z1, z2, label, margin=1.0):
    """
    计算对比损失
    label=1 表示相似，label=0 表示不相似
    """
    dist = F.pairwise_distance(z1, z2)
    loss = label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)
    return loss.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        return contrastive_loss(z1, z2, label, self.margin)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights
)

# ---------------------- 模型定义 ----------------------
class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=256, backbone="resnet34", pretrained=True):
        super().__init__()

        # 使用新版 weights 参数，替代 pretrained=True
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        else:
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet34(weights=weights)

        # 去掉分类层，保留卷积特征提取部分
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        in_dim = base.fc.in_features
        self.fc = nn.Linear(in_dim, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# ---------------------- 对比损失 (类形式) ----------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, y):
        pos = y * dist.pow(2)
        neg = (1 - y) * F.relu(self.margin - dist).pow(2)
        return 0.5 * (pos + neg).mean()

# ---------------------- 对比损失 (函数形式) ----------------------
def contrastive_loss(e1, e2, y, margin=1.0):
    """
    简洁函数形式的对比损失：
    - e1, e2: (B, D) 的 embedding
    - y: (B,) 1=同类, 0=异类
    """
    dist = torch.norm(e1 - e2, p=2, dim=1)
    pos = y * dist.pow(2)
    neg = (1 - y) * F.relu(margin - dist).pow(2)
    return 0.5 * (pos + neg).mean()

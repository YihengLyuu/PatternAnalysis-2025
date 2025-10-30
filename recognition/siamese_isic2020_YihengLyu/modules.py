import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------- 模型 ----------------------
class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=256, backbone="resnet34", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            base = models.resnet50(pretrained=pretrained)
        else:
            base = models.resnet34(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        in_dim = base.fc.in_features
        self.fc = nn.Linear(in_dim, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# ---------------------- 对比损失 (类) ----------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, y):
        pos = y * dist.pow(2)
        neg = (1 - y) * F.relu(self.margin - dist).pow(2)
        return 0.5 * (pos + neg).mean()

# ---------------------- 对比损失 (函数) ----------------------
def contrastive_loss(e1, e2, y, margin=1.0):
    dist = torch.norm(e1 - e2, p=2, dim=1)
    pos = y * dist.pow(2)
    neg = (1 - y) * F.relu(margin - dist).pow(2)
    return 0.5 * (pos + neg).mean()

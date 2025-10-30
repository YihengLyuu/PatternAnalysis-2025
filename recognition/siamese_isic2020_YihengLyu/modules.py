import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def _make_backbone(name: str, pretrained: bool):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = 512
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = 512
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    features = nn.Sequential(*list(m.children())[:-1])  # remove FC -> (B, C, 1, 1)
    return features, feat_dim

class EmbeddingNet(nn.Module):
    """
    Flexible backbone -> GAP -> FC -> BN -> L2-normalized embedding
    """
    def __init__(self, embedding_dim: int = 256, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.features, in_dim = _make_backbone(backbone, pretrained)
        self.fc = nn.Linear(in_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)   # (B, C)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim: int = 256, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.backbone = EmbeddingNet(embedding_dim=embedding_dim, backbone=backbone, pretrained=pretrained)

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        dist = torch.norm(z1 - z2, p=2, dim=1)
        return z1, z2, dist

class ContrastiveLoss(nn.Module):
    """
    y=1(同类) -> 惩罚距离; y=0(异类) -> 惩罚 margin 内的距离
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        pos = label * dist.pow(2)
        neg = (1 - label) * torch.relu(self.margin - dist).pow(2)
        return 0.5 * (pos + neg).mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNet(nn.Module):
    """
    Backbone: ResNet18 (ImageNet 预训练)，去掉分类头 -> 512-d 全局平均池化向量
    再接一层全连接->embedding_dim，L2 normalize 便于余弦/欧式度量。
    """
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
        self.fc = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)              # (B,512,1,1)
        x = x.flatten(1)                  # (B,512)
        x = self.fc(x)                    # (B,emb)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)    # L2-normalized embedding
        return x

class SiameseNet(nn.Module):
    """
    两张图分别过同一个 EmbeddingNet，返回它们的嵌入与距离。
    """
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.backbone = EmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained)

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        # 欧式距离
        dist = torch.norm(z1 - z2, p=2, dim=1)
        return z1, z2, dist

class ContrastiveLoss(nn.Module):
    """
    Hadsell et al., 2006 对比损失:
    y=1(同类) -> 惩罚距离; y=0(异类) -> 惩罚 margin 以内的距离
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        # label: 1=同类, 0=异类
        pos = label * dist.pow(2)
        neg = (1 - label) * F.relu(self.margin - dist).pow(2)
        loss = 0.5 * (pos + neg).mean()
        return loss

# @edu:student-assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os

from dataset import ISIC2020Singles, eval_transforms
from modules import SiameseNet


# ============================================================
# 1) 生成嵌入（支持 TTA）
# ============================================================
@torch.no_grad()
def _embed_with_tta(model, loader, device, tta=True, desc="Embed"):
    """
    通过模型生成图像的特征嵌入；可选水平翻转 TTA。
    返回：(embeddings, labels)
    """
    model.eval()
    feats, labels = [], []
    pbar = tqdm(loader, desc=desc, ncols=100)

    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)

        if tta:
            # 水平翻转 TTA
            x_flip = torch.flip(x, dims=[3])
            z1 = model(x)       # 使用 model()，包含 backbone + fc + L2 normalize
            z2 = model(x_flip)
            z = (z1 + z2) / 2.0
        else:
            z = model(x)

        feats.append(z.cpu())
        labels.extend(y.numpy().tolist())

    feats = torch.cat(feats, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return feats, labels


# ============================================================
# 2) 线性探针评估
# ============================================================
def evaluate_linear_probe(
    checkpoint_path,
    csv_path,
    images_dir,
    batch_size=32,
    img_size=192,
    device=None,
    outdir="runs_eval",
    tta=True,
):
    """
    使用 Siamese 模型提取特征并训练线性探针进行分类评估。
    """
    os.makedirs(outdir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 载入模型 ----
    print(f"[Load] {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = SiameseNet(backbone="resnet34", embed_dim=256)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    # ---- 载入数据 ----
    dataset = ISIC2020Singles(csv_path, images_dir, transform=eval_transforms(img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---- 生成特征 ----
    feats, labels = _embed_with_tta(model, loader, device, tta=tta)

    X = feats.numpy()
    y = labels.numpy().astype(int)

    # ---- 划分 train/test ----
    n = len(X)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(n * 0.8)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # ---- 线性探针 ----
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # ---- 指标 ----
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[Eval] ACC={acc:.4f} | ROC-AUC={roc:.4f} | F1={f1:.4f}")
    print("[Confusion Matrix]")
    print(cm)

    # ---- 绘制混淆矩阵 ----
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Linear Probe Confusion Matrix")
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()

    # ---- 保存结果 ----
    np.savez(os.path.join(outdir, "eval_results.npz"), acc=acc, roc=roc, f1=f1, cm=cm)

    print(f"[Saved] Results → {outdir}")
    return acc, roc, f1


# ============================================================
# 3) CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Siamese ISIC2020 embeddings.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--outdir", type=str, default="runs_eval")
    parser.add_argument("--no_tta", action="store_true")

    args = parser.parse_args()

    evaluate_linear_probe(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        images_dir=args.images,
        batch_size=args.batch_size,
        img_size=args.img_size,
        outdir=args.outdir,
        tta=not args.no_tta,
    )

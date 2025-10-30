# predict.py
# @edu:student-assignment
# Robust eval with class-weighted probe + optimal threshold search

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from tqdm import tqdm
from dataset import ISIC2020Singles, eval_transforms
from modules import SiameseNet


def strip_module_prefix(state):
    if not isinstance(state, dict):
        return state
    return { (k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items() }

def load_state_flex(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    return strip_module_prefix(state)

@torch.no_grad()
def _embed_with_tta(model, loader, device, tta=True, desc="Embed"):
    model.eval()
    feats, labels = [], []
    pbar = tqdm(loader, desc=desc, ncols=100)
    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        if tta:
            x_flip = torch.flip(x, dims=[3])
            z = (model(x) + model(x_flip)) / 2.0  # final embeddings
        else:
            z = model(x)
        feats.append(z.cpu())
        labels.extend(y.numpy().tolist())
    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.float32)

def evaluate_linear_probe(model, records, images_dir, device, batch_size=64, tta=True):
    ds = ISIC2020Singles(records, images_dir, transform=eval_transforms)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 1) 提取嵌入
    feats, labels = _embed_with_tta(model, dl, device, tta=tta, desc="Embed")
    n_pos = int(labels.sum().item())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0:
        raise RuntimeError("No positive samples found. Check CSV 'target' column.")

    # 2) 训练线性探针（类不平衡加权）
    probe = nn.Linear(feats.shape[1], 1).to(device)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    print("[Train linear probe] (pos_weight = {:.1f})".format(pos_weight.item()))
    X = feats.to(device)
    y = labels.to(device)
    for e in range(15):  # 训练稍长一点
        probe.train()
        opt.zero_grad()
        logits = probe(X).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        print(f"  Epoch {e+1}/15 loss={loss.item():.4f}")

    # 3) 计算概率与AUC
    probe.eval()
    with torch.no_grad():
        logits = probe(X).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
    y_true = labels.numpy().astype(int)

    auc = roc_auc_score(y_true, probs)

    # 4) 固定阈值0.5的结果（与之前可对比）
    preds_050 = (probs >= 0.5).astype(int)
    acc_050 = accuracy_score(y_true, preds_050)
    f1_050  = f1_score(y_true, preds_050, zero_division=0)
    cm_050  = confusion_matrix(y_true, preds_050)

    # 5) 搜索最佳阈值（最大化 F1，如果需要也可以换成 Youden J）
    precisions, recalls, ths_pr = precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
    best_idx = int(np.nanargmax(f1s))
    best_thr = ths_pr[max(best_idx-1, 0)] if best_idx < len(ths_pr) else 0.5  # 对齐返回长度
    preds_best = (probs >= best_thr).astype(int)
    acc_best = accuracy_score(y_true, preds_best)
    f1_best  = f1_score(y_true, preds_best, zero_division=0)
    cm_best  = confusion_matrix(y_true, preds_best)

    print("\n[Results — fixed threshold 0.5]")
    print(f"Accuracy: {acc_050:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"F1 Score: {f1_050:.4f}")
    print("Confusion Matrix:\n", cm_050)

    print("\n[Results — optimal threshold for F1]")
    print(f"Best Thr: {best_thr:.4f}")
    print(f"Accuracy: {acc_best:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"F1 Score: {f1_best:.4f}")
    print("Confusion Matrix:\n", cm_best)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "--weights", dest="ckpt", type=str, required=True,
                    help="Path to model weights (.pt)")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--backbone", type=str, default="resnet34",
                    choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--embed_dim", type=int, default=256)
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        records = [r for r in reader]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Load] {args.ckpt}")
    model = SiameseNet(backbone=args.backbone, embed_dim=args.embed_dim).to(device)
    state = load_state_flex(args.ckpt, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warn] Missing keys: {missing}\n[Warn] Unexpected keys: {unexpected}")

    evaluate_linear_probe(
        model=model,
        records=records,
        images_dir=args.images,
        device=device,
        batch_size=args.batch_size,
        tta=not args.no_tta,
    )

if __name__ == "__main__":
    main()

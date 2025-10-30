import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
import pandas as pd

from dataset import ISIC2020Singles, eval_transforms
from modules import SiameseNet


@torch.no_grad()
def _embed_with_tta(model, loader, device, tta=True):
    feats, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        if tta:
            x_flip = torch.flip(x, dims=[3])
            z1 = model.backbone(x)
            z2 = model.backbone(x_flip)
            z = (z1 + z2) / 2.0
        else:
            z = model.backbone(x)
        feats.append(z.cpu())
        labels.extend(y.numpy().tolist())
    return torch.cat(feats), torch.tensor(labels, dtype=torch.float32)


@torch.no_grad()
def evaluate_linear_probe(ckpt_path: str,
                          csv_path: str,
                          images_dir: str,
                          img_size: int = 192,
                          batch_size: int = 64,
                          seed: int = 2025,
                          tta: bool = True,
                          stratify: bool = True,
                          test_csv: str | None = None,
                          use_pos_weight: bool = True,
                          metrics_out: str | None = None):
    """
    冻结 backbone，在线性层上做评估，输出多指标。
    - 若提供 test_csv：仅在给定测试集上评估；其余样本用于训练线性头
      test_csv 需要包含列: image_name, target
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device.type.upper()}  |  ckpt={ckpt_path}  |  img_size={img_size}  |  TTA={tta}")

    # 1) 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    args_ckpt = ckpt.get("args", {})
    embed_dim = args_ckpt.get("embed_dim", 256)
    backbone = args_ckpt.get("backbone", "resnet34")

    model = SiameseNet(embedding_dim=embed_dim, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # 2) 构建 Train/Test 划分
    if test_csv is None:
        df = pd.read_csv(csv_path)
        df["image_name"] = (df["image_name"].astype(str)
                            .str.replace(".jpg", "", regex=False)
                            .str.replace(".png", "", regex=False))
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=seed,
            stratify=df["target"] if stratify else None
        )
        ids_train = train_df["image_name"].tolist()
        ids_test = test_df["image_name"].tolist()
    else:
        # 指定固定测试集：test 用 test_csv，train 用 csv_path 剩余
        df_all = pd.read_csv(csv_path)
        df_all["image_name"] = (df_all["image_name"].astype(str)
                                .str.replace(".jpg", "", regex=False)
                                .str.replace(".png", "", regex=False))
        df_test = pd.read_csv(test_csv)
        df_test["image_name"] = (df_test["image_name"].astype(str)
                                 .str.replace(".jpg", "", regex=False)
                                 .str.replace(".png", "", regex=False))
        test_set_names = set(df_test["image_name"].tolist())
        ids_test = list(test_set_names)
        ids_train = df_all.loc[~df_all["image_name"].isin(test_set_names), "image_name"].tolist()

    ds_train = ISIC2020Singles(csv_path, images_dir,
                               transform=eval_transforms(img_size),
                               ids_subset=ids_train)
    ds_test = ISIC2020Singles(csv_path, images_dir,
                              transform=eval_transforms(img_size),
                              ids_subset=ids_test)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # 3) 提取嵌入
    Xtr, ytr = _embed_with_tta(model, loader_train, device, tta=tta)
    Xte, yte = _embed_with_tta(model, loader_test, device, tta=tta)

    # 4) 训练 Logistic（可选类别加权，提高阳性召回）
    W = torch.zeros(Xtr.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=5e-3, weight_decay=1e-4)
    # 计算 pos_weight：neg/pos（避免除零）
    if use_pos_weight:
        pos = float(ytr.sum().item())
        neg = float(ytr.shape[0] - pos)
        pos_weight = torch.tensor(max(1.0, neg / (pos + 1e-6)))
    else:
        pos_weight = None

    for _ in range(400):
        logits = Xtr @ W + b
        if pos_weight is None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(1), ytr
            )
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(1), ytr, pos_weight=pos_weight
            )
        opt.zero_grad(); loss.backward(); opt.step()

    # 5) 评估多指标
    logits = Xte @ W + b
    proba = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
    pred = (proba > 0.5).astype("float32")
    y_true = yte.cpu().numpy()

    acc = accuracy_score(y_true, pred)
    bacc = balanced_accuracy_score(y_true, pred)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")  # 万一测试集里恰好某一类为0，AUROC没法算
    f1 = f1_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)       # 阳性召回
    pre = precision_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    print(f"[Linear-probe Metrics] acc={acc:.4f}  bacc={bacc:.4f}  auc={auc:.4f}  f1={f1:.4f}  recall_pos={rec:.4f}  precision_pos={pre:.4f}")
    print(f"[Confusion] TN={tn} FP={fp} FN={fn} TP={tp}")

    # 6) 可选保存指标到 CSV
    if metrics_out:
        os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
        dfm = pd.DataFrame([{
            "ckpt": ckpt_path, "img_size": img_size, "tta": int(tta),
            "acc": acc, "balanced_acc": bacc, "auroc": auc, "f1": f1,
            "recall_pos": rec, "precision_pos": pre,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "use_pos_weight": int(use_pos_weight), "stratify": int(stratify),
            "seed": seed
        }])
        if os.path.exists(metrics_out):
            dfm.to_csv(metrics_out, mode="a", header=False, index=False)
        else:
            dfm.to_csv(metrics_out, index=False)

    # 为与旧日志兼容，保留原始提示
    print(f"[Linear-probe Accuracy] {acc:.4f} on 80/20 split using {ckpt_path}")
    return {
        "acc": acc, "balanced_acc": bacc, "auroc": auc, "f1": f1,
        "recall_pos": rec, "precision_pos": pre, "cm": (tn, fp, fn, tp)
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--csv", type=str, default="ISIC_2020_Training_GroundTruth.csv")
    ap.add_argument("--images", type=str, default="train")
    ap.add_argument("--img_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--no_tta", action="store_true", help="关闭 TTA")
    ap.add_argument("--no_stratify", action="store_true", help="关闭分层划分")
    ap.add_argument("--test_csv", type=str, default=None, help="可选：固定测试集 CSV（含 image_name,target）")
    ap.add_argument("--no_pos_weight", action="store_true", help="评估线性探针时不使用类别加权")
    ap.add_argument("--metrics_out", type=str, default=None, help="将指标追加写入该 CSV 路径")

    args = ap.parse_args()
    evaluate_linear_probe(
        ckpt_path=args.ckpt,
        csv_path=args.csv,
        images_dir=args.images,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        tta=not args.no_tta,
        stratify=not args.no_stratify,
        test_csv=args.test_csv,
        use_pos_weight=not args.no_pos_weight,
        metrics_out=args.metrics_out
    )

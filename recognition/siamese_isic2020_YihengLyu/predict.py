import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.manifold import TSNE   # NEW: for embedding visualisation

from dataset import ISIC2020Singles, eval_transforms
from modules import SiameseNet


# ===============================
# Helper: state dict loading
# ===============================
def strip_module_prefix(state):
    if not isinstance(state, dict):
        return state
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


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


# ===============================
# Embedding extraction (with TTA)
# ===============================
@torch.no_grad()
def embed_split(model, records, images_dir, device, batch_size=64,
                tta=True, desc="Embed"):
    ds = ISIC2020Singles(records, images_dir, transform=eval_transforms)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    feats, labels = [], []
    pbar = tqdm(dl, desc=desc, ncols=100)
    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        if tta:
            x_flip = torch.flip(x, dims=[3])
            z = (model(x) + model(x_flip)) / 2.0
        else:
            z = model(x)
        feats.append(z.cpu())
        labels.extend(y.numpy().tolist())

    feats = torch.cat(feats, dim=0)           # [N, D]
    labels = torch.tensor(labels, dtype=torch.float32)
    return feats, labels


# ===============================
# Linear probe training
# ===============================
def train_linear_probe(feats_train, labels_train, device, epochs=15, lr=1e-3):
    X = feats_train.to(device)
    y = labels_train.to(device)

    n_pos = int(y.sum().item())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0:
        raise RuntimeError("No positive samples found in TRAIN split.")
    pos_weight = torch.tensor([n_neg / max(1, n_pos)],
                              dtype=torch.float32, device=device)

    probe = nn.Linear(X.shape[1], 1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    print(f"[Train linear probe] (pos_weight = {pos_weight.item():.1f})")
    for e in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(X).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        print(f"  Epoch {e+1}/{epochs} loss={loss.item():.4f}")

    return probe


# ===============================
# Evaluation on one split
# ===============================
def evaluate_split(probe, feats, labels, split_name, out_fh=None):
    probe.eval()
    X = feats.to(probe.weight.device)
    y = labels.cpu().numpy().astype(int)

    with torch.no_grad():
        logits = probe(X).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    auc = roc_auc_score(y, probs)

    # fixed threshold 0.5
    preds_050 = (probs >= 0.5).astype(int)
    acc_050 = accuracy_score(y, preds_050)
    f1_050 = f1_score(y, preds_050, zero_division=0)
    cm_050 = confusion_matrix(y, preds_050)

    # best F1 threshold
    precisions, recalls, ths_pr = precision_recall_curve(y, probs)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
    best_idx = int(np.nanargmax(f1s))
    best_thr = ths_pr[max(best_idx - 1, 0)] if best_idx < len(ths_pr) else 0.5
    preds_best = (probs >= best_thr).astype(int)
    acc_best = accuracy_score(y, preds_best)
    f1_best = f1_score(y, preds_best, zero_division=0)
    cm_best = confusion_matrix(y, preds_best)

    def _print_and_write(line=""):
        print(line)
        if out_fh is not None:
            out_fh.write(line + "\n")

    _print_and_write(f"\n[Results — {split_name}]")
    _print_and_write(f"ROC-AUC : {auc:.4f}")
    _print_and_write(f"Accuracy (thr=0.5): {acc_050:.4f}")
    _print_and_write(f"F1       (thr=0.5): {f1_050:.4f}")
    _print_and_write(f"Confusion (thr=0.5):\n{cm_050}")
    _print_and_write(f"Best F1 threshold : {best_thr:.4f}")
    _print_and_write(f"Accuracy (best F1): {acc_best:.4f}")
    _print_and_write(f"F1       (best F1): {f1_best:.4f}")
    _print_and_write(f"Confusion (best F1):\n{cm_best}")

    return {
        "auc": auc,
        "acc_050": acc_050,
        "f1_050": f1_050,
        "cm_050": cm_050,
        "best_thr": best_thr,
        "acc_best": acc_best,
        "f1_best": f1_best,
        "cm_best": cm_best,
        "probs": probs,
        "y_true": y,
    }


# ===============================
# t-SNE visualisation
# ===============================
def visualize_tsne(feats, labels, outdir, split_name="test", max_points=2000):
    os.makedirs(outdir, exist_ok=True)

    X = feats.cpu().numpy()
    y = labels.cpu().numpy().astype(int)

    n = len(X)
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"[t-SNE] running on {len(X)} points for split '{split_name}'...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    # 简单双类配色：0 = benign, 1 = melanoma
    for cls, name in [(0, "benign"), (1, "melanoma")]:
        mask = (y == cls)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=6,
            alpha=0.6,
            label=name
        )
    plt.title(f"t-SNE of embeddings ({split_name} split)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    out_path = os.path.join(outdir, f"tsne_{split_name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[t-SNE] saved to {out_path}")


# ===============================
# Entry point
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "--weights", dest="ckpt", type=str, required=True,
                    help="Path to model weights (.pt)")

    # 兼容旧接口：如果你只给 --csv / --images，就会被当成 both train & test
    ap.add_argument("--csv", type=str, default=None,
                    help="(legacy) CSV used for both train and test")
    ap.add_argument("--images", type=str, default=None,
                    help="(legacy) image dir used for both train and test")

    # 新接口：明确的 train / test
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--train_images", type=str, default=None)
    ap.add_argument("--test_csv", type=str, default=None,
                    help="CSV for the 1,000-image test split")
    ap.add_argument("--test_images", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--outdir", type=str, default="eval_results")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--backbone", type=str, default="resnet34",
                    choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--tsne_points", type=int, default=2000,
                    help="max points used for t-SNE")
    args = ap.parse_args()

    # --------- resolve train/test paths (兼容旧参数) ----------
    if args.train_csv is None or args.train_images is None:
        # 用旧接口
        if args.csv is None or args.images is None:
            raise ValueError("Either (--train_csv & --train_images) or (--csv & --images) must be provided.")
        args.train_csv = args.csv
        args.train_images = args.images

    if args.test_csv is None or args.test_images is None:
        # 如果没给 test，就默认 test=train（不推荐，但保持兼容）
        args.test_csv = args.train_csv
        args.test_images = args.train_images

    # --------- load CSV records ----------
    def load_records(path):
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            return [r for r in reader if r]

    train_records = load_records(args.train_csv)
    test_records = load_records(args.test_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Load] checkpoint: {args.ckpt}")
    model = SiameseNet(backbone=args.backbone, embed_dim=args.embed_dim).to(device)
    state = load_state_flex(args.ckpt, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warn] Missing keys: {missing}\n[Warn] Unexpected keys: {unexpected}")

    # --------- extract embeddings ----------
    feats_train, labels_train = embed_split(
        model, train_records, args.train_images,
        device=device, batch_size=args.batch_size,
        tta=not args.no_tta, desc="Embed train"
    )
    feats_test, labels_test = embed_split(
        model, test_records, args.test_images,
        device=device, batch_size=args.batch_size,
        tta=not args.no_tta, desc="Embed test"
    )

    # --------- train linear probe on TRAIN only ----------
    probe = train_linear_probe(feats_train, labels_train, device=device, epochs=15, lr=1e-3)

    # --------- evaluate on TRAIN & TEST ----------
    os.makedirs(args.outdir, exist_ok=True)
    txt_path = os.path.join(args.outdir, "eval_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        res_train = evaluate_split(probe, feats_train, labels_train, "TRAIN", out_fh=f)
        res_test = evaluate_split(probe, feats_test, labels_test, "TEST (1,000 images)", out_fh=f)
    print(f"\n✅ Evaluation results saved to: {txt_path}")

    # --------- ROC curve on TEST ----------
    fpr, tpr, _ = roc_curve(res_test["y_true"], res_test["probs"])
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {res_test['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve – TEST split")
    plt.legend(loc="lower right")
    roc_path = os.path.join(args.outdir, "roc_curve_test.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"📈 Test ROC curve saved to: {roc_path}")

    # --------- t-SNE on TEST embeddings ----------
    visualize_tsne(feats_test, labels_test, args.outdir,
                   split_name="test", max_points=args.tsne_points)


if __name__ == "__main__":
    main()

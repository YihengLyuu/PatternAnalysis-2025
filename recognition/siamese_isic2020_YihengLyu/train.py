import os
import csv
import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import ISIC2020Pairs, train_transforms, eval_transforms
from modules import SiameseNet, ContrastiveLoss


# ===============================
# Utility: save loss/accuracy curves
# ===============================
def save_curves(history, outdir):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_pair_acc"], label="val_pair_acc")
    plt.legend()
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.savefig(os.path.join(outdir, "training_curves.png"))
    plt.close()


# ===============================
# Validation function
# ===============================
def validate(model, loader, device, dist_threshold=0.5):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            f1, f2 = model(x1, x2)
            dist = F.pairwise_distance(f1, f2)
            pred = (dist < dist_threshold).float()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / max(1, total)


# ===============================
# CSV utilities
# ===============================
def read_csv_with_header(csv_path):
    """Return (header, rows) where both are list[str] lists."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        all_rows = [r for r in reader if r]  # drop empty lines
    if not all_rows:
        raise ValueError("CSV is empty.")
    header = all_rows[0]
    rows = all_rows[1:] if header and "image_name" in header else all_rows
    if header and "image_name" not in header:
        # fallback if no header present
        header = ["image_name", "patient_id", "sex", "age_approx",
                  "anatom_site_general_challenge", "diagnosis",
                  "benign_malignant", "target"][:len(all_rows[0])]
        rows = all_rows
    return header, rows


def get_col_indices(header):
    """Find required column indices."""
    try:
        idx_img = header.index("image_name")
        idx_pid = header.index("patient_id")
        idx_tgt = header.index("target")
    except ValueError as e:
        raise ValueError(f"CSV header missing column: {e}. "
                         f"Expected at least ['image_name','patient_id','target']. Got: {header}")
    return idx_img, idx_pid, idx_tgt


def grouped_split_by_patient(rows, header, val_ratio=0.2, seed=42, stratify=True):
    """
       Patient-level train/val split to prevent data leakage.

       - We first group all rows by patient_id.
       - Each patient is assigned a pseudo-label = max(target) within that patient.
       - We then perform a stratified split at the patient level (positive / negative),
         so that no patient appears in both train and val.
       - Pair sampling (ISIC2020Pairs) is only applied *after* this split,
         which ensures that pairs never mix patients across train/val.
       """
    idx_img, idx_pid, idx_tgt = get_col_indices(header)

    # Group rows by patient_id
    by_pid = defaultdict(list)
    for r in rows:
        if len(r) <= max(idx_img, idx_pid, idx_tgt):
            continue
        pid = r[idx_pid]
        by_pid[pid].append(r)

    # Build patient labels (any positive within patient => positive)
    patient_items = []
    for pid, rlist in by_pid.items():
        has_pos = any((str(rr[idx_tgt]).strip() == "1") for rr in rlist)
        patient_items.append((pid, rlist, 1 if has_pos else 0))

    # Shuffle for reproducibility
    rnd = random.Random(seed)
    if stratify:
        pos = [item for item in patient_items if item[2] == 1]
        neg = [item for item in patient_items if item[2] == 0]
        rnd.shuffle(pos)
        rnd.shuffle(neg)
        n_pos_val = max(1, int(round(len(pos) * val_ratio)))
        n_neg_val = max(1, int(round(len(neg) * val_ratio)))
        val_items = pos[:n_pos_val] + neg[:n_neg_val]
        train_items = pos[n_pos_val:] + neg[n_neg_val:]
        rnd.shuffle(train_items)
        rnd.shuffle(val_items)
    else:
        rnd.shuffle(patient_items)
        n_val = max(1, int(round(len(patient_items) * val_ratio)))
        val_items = patient_items[:n_val]
        train_items = patient_items[n_val:]

    # Flatten back to row lists
    train_rows = [r for (_, rlist, _) in train_items for r in rlist]
    val_rows = [r for (_, rlist, _) in val_items for r in rlist]

    # Log stats
    n_train_pos = sum(int(r[idx_tgt]) for r in train_rows if len(r) > idx_tgt and str(r[idx_tgt]).isdigit())
    n_train = len(train_rows)
    n_val_pos = sum(int(r[idx_tgt]) for r in val_rows if len(r) > idx_tgt and str(r[idx_tgt]).isdigit())
    n_val = len(val_rows)

    print(f"[Split] patients={len(patient_items)}  "
          f"train_rows={n_train} (pos={n_train_pos})  "
          f"val_rows={n_val} (pos={n_val_pos})")
    return train_rows, val_rows


# ===============================
# Main training function
# ===============================
def train_siamese(args):
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load and split CSV by patient_id
    header, rows = read_csv_with_header(args.csv)
    train_records, val_records = grouped_split_by_patient(
        rows, header, val_ratio=0.2, seed=42, stratify=True
    )

    # 2) Datasets and Loaders
    train_ds = ISIC2020Pairs(train_records, args.images,
                             pairs_per_epoch=args.pairs_per_epoch,
                             transform=train_transforms)
    val_ds = ISIC2020Pairs(val_records, args.images,
                           pairs_per_epoch=400, transform=eval_transforms)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # 3) Model / Optimizer / Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet(backbone=args.backbone, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = ContrastiveLoss(margin=1.0)

    history = {"train_loss": [], "val_pair_acc": []}
    best_acc = 0.0
    best_epoch = -1

    # 4) Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        for x1, x2, y in pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.float().to(device)
            f1, f2 = model(x1, x2)
            loss = loss_fn(f1, f2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = validate(model, val_dl, device, dist_threshold=0.5)
        train_loss = sum(losses) / max(1, len(losses))
        history["train_loss"].append(train_loss)
        history["val_pair_acc"].append(val_acc)
        print(f"[Epoch {epoch}/{args.epochs}] loss={train_loss:.4f}, val_pair_acc={val_acc:.4f}")

        # Save model every epoch
        ckpt_path = os.path.join(args.outdir, f"siamese_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))

    # 5) Save results
    save_curves(history, args.outdir)
    print(f"Training completed. Best val_pair_acc={best_acc:.4f} (epoch {best_epoch})")


# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SiameseNet on ISIC2020 dataset (patient-level split)")

    parser.add_argument("--csv", type=str, required=True, help="Path to ISIC_2020_Training_GroundTruth.csv")
    parser.add_argument("--images", type=str, required=True, help="Path to training image folder")
    parser.add_argument("--outdir", type=str, default="runs", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--pairs_per_epoch", type=int, default=2000, help="Number of pairs per epoch")
    parser.add_argument("--img_size", type=int, default=192, help="Image size (square)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--prefetch", type=int, default=4, help="Prefetch buffer size")
    parser.add_argument("--backbone", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="Backbone network for SiameseNet")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")

    args = parser.parse_args()
    train_siamese(args)

import os
import argparse
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import ISIC2020Pairs
from modules import SiameseNet, contrastive_loss

def save_curves(losses, accs, outdir):
    epochs = range(1, len(losses) + 1)
    plt.figure()
    plt.plot(epochs, losses, label="train_loss", marker='o')
    plt.plot(epochs, accs, label="val_pair_acc", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Siamese Training Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "training_curves.png"))
    plt.close()

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (x1, x2, y) in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            e1, e2 = model(x1), model(x2)
            dist = torch.norm(e1 - e2, dim=1)
            pred = (dist < 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0

def train_siamese(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device.type.upper()} | AMP={args.amp_dtype} | backbone={args.backbone} | img_size={args.img_size} | embed_dim={args.embed_dim}")

    # datasets / loaders
    train_ds = ISIC2020Pairs(args.csv, args.images, "train", args.pairs_per_epoch, args.img_size)
    val_ds   = ISIC2020Pairs(args.csv, args.images, "val",   int(args.pairs_per_epoch*0.2), args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # model / opt
    model = SiameseNet(args.embed_dim, args.backbone).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda" and args.amp_dtype=="bf16"))

    # EMA
    ema_model = SiameseNet(args.embed_dim, args.backbone).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = args.ema_decay

    # logs
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","loss","val_acc","lr"])

    best_acc = -1.0
    best_ckpt_path = None
    losses, accs = [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        for x1, x2, y in pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type=="cuda" and args.amp_dtype=="bf16")):
                e1, e2 = model(x1), model(x2)
                loss = contrastive_loss(e1, e2, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(ema_decay).add_(p.data, alpha=1-ema_decay)

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        val_acc  = evaluate(ema_model, val_loader, device)
        lr = scheduler.get_last_lr()[0]

        print(f"[Epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  val_pair_acc={val_acc:.4f}  lr={lr:.6f}")
        losses.append(avg_loss); accs.append(val_acc)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_loss, val_acc, lr])

        # ✅ 每轮都保存
        this_ckpt = os.path.join(args.outdir, f"siamese_epoch{epoch+1}.pt")
        torch.save({"state_dict": ema_model.state_dict(), "args": vars(args)}, this_ckpt)

        # ✅ 记录最佳
        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt_path = this_ckpt
            torch.save({"state_dict": ema_model.state_dict(), "args": vars(args)},
                       os.path.join(args.outdir, "best.pt"))

    save_curves(losses, accs, args.outdir)
    print(f"Saved curves to {args.outdir}/training_curves.png")
    print(f"[Best] val_pair_acc={best_acc:.4f}  ckpt={os.path.basename(best_ckpt_path) if best_ckpt_path else 'N/A'}")

    if args.final_eval and best_ckpt_path:
        from predict import evaluate_linear_probe
        print(f"[Final Eval] Using checkpoint: {best_ckpt_path} (best.pt)")
        evaluate_linear_probe(best_ckpt_path, args.csv, args.images, args.img_size)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="ISIC_2020_Training_GroundTruth.csv")
    ap.add_argument("--images", type=str, default="train")
    ap.add_argument("--outdir", type=str, default="runs_test")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--pairs_per_epoch", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--backbone", type=str, default="resnet34")
    ap.add_argument("--img_size", type=int, default=192)
    ap.add_argument("--amp_dtype", type=str, default="bf16")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--final_eval", action="store_true")
    args = ap.parse_args()
    train_siamese(args)

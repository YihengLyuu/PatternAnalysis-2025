import argparse, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torchvision.ops import misc as tv_misc
import matplotlib.pyplot as plt

from dataset import ISIC2020Pairs, ISIC2020Singles, default_transforms, eval_transforms
from modules import SiameseNet, ContrastiveLoss

def accuracy_from_pairs(dist, y, margin=1.0):
    """
    用阈值将距离变为“同类/异类”判别：dist < t 认为同类。
    简单取 t = margin/2，实际可在验证集上寻优。
    """
    t = margin / 2.0
    pred_same = (dist < t).float()
    return (pred_same == y).float().mean().item()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据：拆 train/val
    full_singles = ISIC2020Singles(args.csv, args.images, transform=eval_transforms(args.img_size))
    val_size = max(200, int(0.1 * len(full_singles)))
    train_size = len(full_singles) - val_size
    train_subset, val_subset = random_split(full_singles, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(2025))

    # 训练对儿数据集（在线配对）
    train_pairs = ISIC2020Pairs(args.csv, args.images,
                                transform=default_transforms(args.img_size),
                                pairs_per_epoch=args.pairs_per_epoch, seed=2025)

    val_loader_pairs = DataLoader(
        ISIC2020Pairs(args.csv, args.images,
                      transform=eval_transforms(args.img_size),
                      pairs_per_epoch=2000, seed=7),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    train_loader = DataLoader(train_pairs, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # 模型与优化
    model = SiameseNet(embedding_dim=args.embed_dim, pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=args.margin).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.outdir, exist_ok=True)
    history = {"loss": [], "acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = []
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            _, _, dist = model(x1, x2)
            loss = criterion(dist, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss.append(loss.item())

        # 验证一次（基于对儿判别的准确率）
        model.eval()
        with torch.no_grad():
            all_dist, all_y = [], []
            for x1, x2, y in val_loader_pairs:
                x1, x2 = x1.to(device), x2.to(device)
                _, _, dist = model(x1, x2)
                all_dist.append(dist.cpu())
                all_y.append(y)
            all_dist = torch.cat(all_dist)
            all_y = torch.cat(all_y)
            acc = accuracy_from_pairs(all_dist, all_y, margin=args.margin)

        epoch_loss = float(np.mean(running_loss))
        history["loss"].append(epoch_loss)
        history["acc"].append(acc)
        print(f"[Epoch {epoch}/{args.epochs}] loss={epoch_loss:.4f}  val_pair_acc={acc:.4f}")

        # 保存 checkpoint
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "val_pair_acc": acc,
        }, os.path.join(args.outdir, f"siamese_epoch{epoch}.pt"))

    # 画图（loss & pair-acc）
    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["acc"], label="val_pair_acc")
    plt.xlabel("epoch"); plt.legend(); plt.title("Siamese Training")
    fig_path = os.path.join(args.outdir, "training_curves.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved curves to {fig_path}")

    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="ISIC_2020_Training_GroundTruth.csv")
    parser.add_argument("--images", type=str, default="train")
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pairs_per_epoch", type=int, default=20000)
    args = parser.parse_args()
    main(args)

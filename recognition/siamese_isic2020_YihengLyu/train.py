import argparse, os, json, csv, time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import matplotlib.pyplot as plt

# 进度条：如果没装 tqdm，不报错，降级为普通迭代
try:
    from tqdm import tqdm
    def progress(iterable, total=None, desc=""):
        return tqdm(iterable, total=total, desc=desc, ncols=100)
except Exception:
    def progress(iterable, total=None, desc=""):
        return iterable

from dataset import ISIC2020Pairs, ISIC2020Singles, default_transforms, eval_transforms
from modules import SiameseNet, ContrastiveLoss

# 直接复用 predict.py 中的评估函数，避免再开进程
try:
    from predict import evaluate_linear_probe
    HAS_EVAL = True
except Exception:
    HAS_EVAL = False


def accuracy_from_pairs(dist, y, margin=1.0):
    t = margin / 2.0
    pred_same = (dist < t).float()
    return (pred_same == y).float().mean().item()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def append_row_to_csv(csv_path, row_dict, header_order):
    # 若文件不存在，先写表头
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not file_exists:
            w.writeheader()
        w.writerow(row_dict)


def main(args):
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==== 数据 ====
    full_singles = ISIC2020Singles(args.csv, args.images, transform=eval_transforms(args.img_size))
    val_size = max(200, int(0.1 * len(full_singles)))
    train_size = len(full_singles) - val_size
    train_subset, val_subset = random_split(
        full_singles, [train_size, val_size],
        generator=torch.Generator().manual_seed(2025)
    )

    train_pairs = ISIC2020Pairs(
        args.csv, args.images,
        transform=default_transforms(args.img_size),
        pairs_per_epoch=args.pairs_per_epoch, seed=2025
    )

    val_loader_pairs = DataLoader(
        ISIC2020Pairs(args.csv, args.images,
                      transform=eval_transforms(args.img_size),
                      pairs_per_epoch=min(4000, max(2000, args.batch_size * 64)),
                      seed=7),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    train_loader = DataLoader(
        train_pairs, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # ==== 模型 ====
    model = SiameseNet(embedding_dim=args.embed_dim, pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=args.margin).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ensure_dir(args.outdir)
    history = {"loss": [], "acc": []}

    # CSV 日志路径
    metrics_csv = args.log_csv or os.path.join(args.outdir, "metrics.csv")
    header = ["epoch", "train_loss", "val_pair_acc"]

    steps_per_epoch = int(np.ceil(args.pairs_per_epoch / max(1, args.batch_size)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = []

        # ====== 训练带进度条 ======
        it = progress(range(steps_per_epoch), total=steps_per_epoch,
                      desc=f"Epoch {epoch}/{args.epochs}") if args.progress else range(steps_per_epoch)

        # 因为我们使用在线配对的数据集，这里按“步数”迭代而不是直接 for batch in loader
        train_iter = iter(train_loader)
        for _ in it:
            try:
                x1, x2, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x1, x2, y = next(train_iter)

            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            _, _, dist = model(x1, x2)
            loss = criterion(dist, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss.append(loss.item())

            if args.progress and hasattr(it, "set_postfix"):
                it.set_postfix(loss=f"{running_loss[-1]:.4f}")

        # ====== 验证 ======
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
        ckpt_path = os.path.join(args.outdir, f"siamese_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "val_pair_acc": acc,
        }, ckpt_path)

        # ====== 追加写入 CSV ======
        append_row_to_csv(metrics_csv, {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_pair_acc": acc
        }, header)

    # ==== 画图 ====
    plt.figure()
    plt.plot(history["loss"], label="train_loss", marker='o')
    plt.plot(history["acc"], label="val_pair_acc", marker='o')
    plt.xlabel("epoch"); plt.legend(); plt.title("Siamese Training")
    fig_path = os.path.join(args.outdir, "training_curves.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved curves to {fig_path}")

    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ==== 训练后自动评估（linear probe 单图准确率） ====
    if args.final_eval:
        if not HAS_EVAL:
            print("[Final Eval] 未找到 predict.evaluate_linear_probe；请确认 predict.py 在同目录。")
        else:
            last_ckpt = ckpt_path  # 就是上面刚存的最后一个
            print(f"[Final Eval] Using checkpoint: {last_ckpt}")
            acc_final = evaluate_linear_probe(
                ckpt_path=last_ckpt,
                csv=args.csv,
                images=args.images,
                img_size=args.img_size,
                batch_size=args.eval_batch_size
            )
            # 同步写入 CSV
            append_row_to_csv(metrics_csv, {
                "epoch": "final_eval",
                "train_loss": history["loss"][-1],
                "val_pair_acc": history["acc"][-1]
            }, header)
            print(f"[Final Eval] Done. Linear-probe accuracy printed above.")

    dt = time.time() - t0
    print(f"Total wall time: {dt/60:.1f} min")


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
    parser.add_argument("--num_workers", type=int, default=4)

    # 新增：日志/进度/自动评估
    parser.add_argument("--log_csv", type=str, default=None,
                        help="每个 epoch 的指标写到这个 CSV（默认 outdir/metrics.csv）")
    parser.add_argument("--progress", action="store_true",
                        help="显示 tqdm 进度条")
    parser.add_argument("--final_eval", action="store_true",
                        help="训练结束后自动评估 linear-probe accuracy")
    parser.add_argument("--eval_batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)

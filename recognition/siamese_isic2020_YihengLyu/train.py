import argparse, os, json, csv, time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# 进度条
try:
    from tqdm import tqdm
    def progress(iterable, total=None, desc=""): return tqdm(iterable, total=total, desc=desc, ncols=100)
except Exception:
    def progress(iterable, total=None, desc=""): return iterable

from dataset import ISIC2020Pairs, ISIC2020Singles, default_transforms, eval_transforms
from modules import SiameseNet, ContrastiveLoss

try:
    from predict import evaluate_linear_probe
    HAS_EVAL = True
except Exception:
    HAS_EVAL = False


# --------- 简单 EMA 实现 ----------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.detach().clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.data.detach().clone()
            p.data = self.shadow[name].detach().clone()

    def restore(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data = self.backup[name]
        self.backup = {}
# -----------------------------------


def accuracy_from_pairs(dist, y, margin=1.0):
    t = margin / 2.0
    pred_same = (dist < t).float()
    return (pred_same == y).float().mean().item()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def append_row_to_csv(csv_path, row_dict, header_order):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not file_exists: w.writeheader()
        w.writerow(row_dict)


def main(args):
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'

    # ===== 速度开关 =====
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # AMP
    if args.amp == "auto":
        args.amp = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ("fp16" if torch.cuda.is_available() else "off")
    amp_enabled = args.amp != "off" and use_cuda
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    # ===== 打印设备与关键配置 =====
    print(f"[Device] Using: {device.type.upper()} | AMP={args.amp} | backbone={args.backbone} | img_size={args.img_size} | embed_dim={args.embed_dim}")

    # ===== 数据 =====
    full_singles = ISIC2020Singles(args.csv, args.images, transform=eval_transforms(args.img_size))
    val_size = max(200, int(0.1 * len(full_singles)))
    train_size = len(full_singles) - val_size
    train_subset, val_subset = random_split(full_singles, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(2025))

    train_pairs = ISIC2020Pairs(args.csv, args.images,
                                transform=default_transforms(args.img_size),
                                pairs_per_epoch=args.pairs_per_epoch, seed=2025)

    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, persistent_workers=(args.num_workers > 0))
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch

    val_loader_pairs = DataLoader(
        ISIC2020Pairs(args.csv, args.images,
                      transform=eval_transforms(args.img_size),
                      pairs_per_epoch=min(4000, max(2000, args.batch_size * 64)), seed=7),
        shuffle=False, **dl_kwargs
    )
    train_loader = DataLoader(train_pairs, shuffle=True, **dl_kwargs)

    # ===== 模型 =====
    model = SiameseNet(embedding_dim=args.embed_dim, backbone=args.backbone, pretrained=True).to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[Speed] torch.compile 已启用")
        except Exception as e:
            print(f"[Speed] torch.compile 未启用: {e}")

    criterion = ContrastiveLoss(margin=args.margin).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR：warmup + cosine
    total_epochs = args.epochs
    warmup_epochs = max(1, args.warmup_epochs)
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    scheduler = CosineAnnealingLR(optim, T_max=cosine_epochs, eta_min=args.lr * args.cosine_min_ratio)

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    ensure_dir(args.outdir)
    history = {"loss": [], "acc": []}
    metrics_csv = args.log_csv or os.path.join(args.outdir, "metrics.csv")
    header = ["epoch", "train_loss", "val_pair_acc", "lr"]

    steps_per_epoch = int(np.ceil(args.pairs_per_epoch / max(1, args.batch_size)))

    current_epoch = 0
    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = []

        it = progress(range(steps_per_epoch), total=steps_per_epoch,
                      desc=f"Epoch {epoch}/{total_epochs}") if args.progress else range(steps_per_epoch)

        train_iter = iter(train_loader)
        for _ in it:
            try:
                x1, x2, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x1, x2, y = next(train_iter)

            if use_cuda:
                x1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                x2 = x2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                x1 = x1.to(device); x2 = x2.to(device)
            y = y.to(device, non_blocking=True)

            # --- Warmup 学习率 ---
            if epoch <= warmup_epochs:
                warmup_ratio = ( (epoch-1) * steps_per_epoch + _ + 1 ) / (warmup_epochs * steps_per_epoch)
                lr_now = args.lr * warmup_ratio
                for pg in optim.param_groups:
                    pg["lr"] = lr_now

            if amp_enabled:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    _, _, dist = model(x1, x2)
                    loss = criterion(dist, y)
                optim.zero_grad(set_to_none=True)
                if amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
            else:
                _, _, dist = model(x1, x2)
                loss = criterion(dist, y)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            if ema is not None:
                ema.update(model)

            running_loss.append(loss.item())
            if args.progress and hasattr(it, "set_postfix"):
                it.set_postfix(loss=f"{running_loss[-1]:.4f}")

        # --- Cosine 学习率步进（warmup 之后才动） ---
        if epoch > warmup_epochs:
            scheduler.step()

        # ====== 验证（使用 EMA 权重更稳） ======
        model.eval()
        if ema is not None:
            ema.apply_shadow(model)
        with torch.no_grad():
            all_dist, all_y = [], []
            for x1, x2, y in val_loader_pairs:
                if use_cuda:
                    x1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    x2 = x2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                else:
                    x1 = x1.to(device); x2 = x2.to(device)
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    _, _, dist = model(x1, x2)
                all_dist.append(dist.float().cpu())
                all_y.append(y)
            all_dist = torch.cat(all_dist); all_y = torch.cat(all_y)
            acc = accuracy_from_pairs(all_dist, all_y, margin=args.margin)
        if ema is not None:
            ema.restore(model)

        epoch_loss = float(np.mean(running_loss))
        history["loss"].append(epoch_loss)
        history["acc"].append(acc)
        lr_show = optim.param_groups[0]["lr"]
        print(f"[Epoch {epoch}/{total_epochs}] loss={epoch_loss:.4f}  val_pair_acc={acc:.4f}  lr={lr_show:.6f}")

        # 保存 checkpoint（保存 EMA 权重版本）
        ckpt_path = os.path.join(args.outdir, f"siamese_epoch{epoch}.pt")
        if ema is not None:
            ema.apply_shadow(model)
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "args": vars(args), "val_pair_acc": acc}, ckpt_path)
        if ema is not None:
            ema.restore(model)

        append_row_to_csv(metrics_csv, {"epoch": epoch, "train_loss": epoch_loss, "val_pair_acc": acc, "lr": lr_show}, header)

    # ===== 画图 =====
    plt.figure()
    plt.plot(history["loss"], label="train_loss", marker='o')
    plt.plot(history["acc"], label="val_pair_acc", marker='o')
    plt.xlabel("epoch"); plt.legend(); plt.title("Siamese Training")
    fig_path = os.path.join(args.outdir, "training_curves.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved curves to {fig_path}")

    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ===== 训练后自动评估 =====
    if args.final_eval:
        if not HAS_EVAL:
            print("[Final Eval] 未找到 predict.evaluate_linear_probe；请确认 predict.py 在同目录。")
        else:
            last_ckpt = ckpt_path
            print(f"[Final Eval] Using checkpoint: {last_ckpt}")
            _ = evaluate_linear_probe(
                ckpt_path=last_ckpt, csv=args.csv, images=args.images,
                img_size=args.img_size, batch_size=args.eval_batch_size
            )
            append_row_to_csv(metrics_csv, {"epoch": "final_eval",
                                            "train_loss": history["loss"][-1],
                                            "val_pair_acc": history["acc"][-1],
                                            "lr": optim.param_groups[0]["lr"]}, header)

    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="ISIC_2020_Training_GroundTruth.csv")
    parser.add_argument("--images", type=str, default="train")
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18","resnet34","resnet50"])
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cosine_min_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pairs_per_epoch", type=int, default=20000)

    # 性能项
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "fp16", "bf16", "off"])
    parser.add_argument("--compile", action="store_true")

    # 日志/评估
    parser.add_argument("--log_csv", type=str, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--final_eval", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=128)

    args = parser.parse_args()
    main(args)

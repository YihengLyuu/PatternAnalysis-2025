import argparse
import torch
from torch.utils.data import DataLoader
from dataset import ISIC2020Singles, eval_transforms
from modules import SiameseNet

@torch.no_grad()
def evaluate_linear_probe(ckpt_path, csv, images, img_size=192, batch_size=64):
    """
    冻结 backbone，线性探针评估（80/20）。使用简单TTA（原图+水平翻转）的嵌入平均，提升最终 accuracy。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    args_ckpt = ckpt.get("args", {})
    embed_dim = args_ckpt.get("embed_dim", 256)
    backbone = args_ckpt.get("backbone", "resnet34")

    model = SiameseNet(embedding_dim=embed_dim, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    ds = ISIC2020Singles(csv, images, transform=eval_transforms(img_size))
    n = len(ds)
    n_train = int(0.8 * n)
    idx_train = list(range(0, n_train))
    idx_test = list(range(n_train, n))
    ds_train = ISIC2020Singles(csv, images, transform=eval_transforms(img_size),
                               ids_subset=[ds.df.iloc[i]['image_name'] for i in idx_train])
    ds_test  = ISIC2020Singles(csv, images, transform=eval_transforms(img_size),
                               ids_subset=[ds.df.iloc[i]['image_name'] for i in idx_test])

    def embed_dataset(ds_local):
        loader = DataLoader(ds_local, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        feats, labels = [], []
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            # TTA: 原图 + 水平翻转
            x_flip = torch.flip(x, dims=[3])
            z1 = model.backbone(x)
            z2 = model.backbone(x_flip)
            z = (z1 + z2) / 2.0
            feats.append(z.cpu())
            labels.extend(y.numpy().tolist())
        return torch.cat(feats), torch.tensor(labels, dtype=torch.float32)

    Xtr, ytr = embed_dataset(ds_train)
    Xte, yte = embed_dataset(ds_test)

    # 训练一个 logistic 回归（sigmoid）
    W = torch.zeros(Xtr.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=5e-3, weight_decay=1e-4)
    for _ in range(400):
        logits = Xtr @ W + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), ytr)
        opt.zero_grad(); loss.backward(); opt.step()

    logits = Xte @ W + b
    pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
    acc = (pred == yte).float().mean().item()
    print(f"[Linear-probe Accuracy] {acc:.4f} on 80/20 split using {ckpt_path}")
    return acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--csv", type=str, default="ISIC_2020_Training_GroundTruth.csv")
    ap.add_argument("--images", type=str, default="train")
    ap.add_argument("--img_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    evaluate_linear_probe(args.ckpt, args.csv, args.images, args.img_size, args.batch_size)
